#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/merge.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <unistd.h>

#include <chrono>
#include <csetjmp>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "alloc.h"
#include "bindings.h"
#include "dataflow.h"
#include "device_vec.h"
#include "flame.h"
#include "merge.h"
#include "normalize.h"
#include "provenance.h"
#include "table.h"
#include "utils.h"

static int step = 0;

static void *last_ret_value = nullptr;

void *lobster_global_alloc_allocate(size_t size) {
  return lobster_global_allocator().malloc<char>(size);
}
void lobster_global_alloc_deallocate(void *ptr) {
  lobster_global_allocator().destroy<char>((char *)ptr);
}

inline size_t SUBSPACE_SIZE() {
  static std::optional<size_t> trigger = std::nullopt;
  if (!trigger.has_value()) {
    auto *s = std::getenv("SUBSPACE");
    if (s) {
      trigger = std::stoul(s);
    } else {
      trigger = 0;
    }
  }
  return trigger.value();
}

template <typename Prov>
void execute_stratum_device(const Stratum<Prov> &stratum, DynamicIdb<Prov> &idb,
                            Prov &ctx) {
  (void)ctx;

  step += 1;

  TRACE_START(execute_stratum_device);

  bool LOG_STRATUM = std::getenv("LOG_STRATUM") != nullptr;

  if (LOG_STRATUM) {
    std::cout << "stratum: " << stratum << std::endl;
    std::cout << "current idb: " << idb << std::endl;
  }

  // free any memory from previous executions
  DeviceAlloc::singleton_alloc().reset();

  // start timer
  Timer main_timer{"main"};

  main_timer.start();
  int iteration = 0;

  idb.validate();

  // the set of relations that are targets of updates in this stratum
  std::vector<std::string> target_relations;
  for (size_t i = 0; i < stratum.relations().size(); i++) {
    target_relations.push_back(stratum.relations()[i].predicate().to_string());
  }

  // evaluate stable to get the starting facts for this stratum
  hINFO("Evaluating rules on stable facts");
  for (size_t i = 0; i < stratum.updates().size(); i++) {
    TRACE_START(dataflow_evaluate_stable);
    const auto &update = stratum.updates()[i];
    const auto &target_name = update.target().to_string();
    auto &target = idb[target_name];

    hINFO("Running stratum with target: " << target_name);

    auto active_df =
        ActiveDataflow<Prov>::from_dataflow(update.dataflow(), idb);

    auto output = active_df->evaluate_stable(idb, ctx);
    // Clone the output if the active dataflow is just a source so that the
    // mutations caused by incorporate delta don't mess with the database
    if (dynamic_cast<ActiveDataflowSource<Prov> *>(active_df.get()) !=
        nullptr) {
      output = output.clone();
    }

    target.delta().append(output);
  }

  // if the stratum isn't recursive, just running evaluate_stable and
  // normalization is enough
  if (!stratum.is_recursive()) {
    for (auto &&pair : idb.relations()) {
      auto &name = pair.first;
      auto &relation = pair.second;

      if (relation.stable().size() != 0 && relation.delta().size() != 0) {
        PANIC("Relation %s has both stable and delta facts", name.c_str());
      }
      if (relation.delta().size() != 0) {
        relation.stable() =
            normalized(relation.delta(), ctx, DeviceAlloc().singleton_alloc());
        relation.delta().clear();
      }
    }
    return;
  }

  // otherwise, run the fixpoint loop
  idb.incorporate_delta(ctx, DeviceAlloc().singleton_alloc());

  hINFO("Running fixed point evaluation loop");
  while (true) {
    for (size_t i = 0; i < stratum.updates().size(); i++) {
      TRACE_START(dataflow_evaluate);
      const auto &update = stratum.updates()[i];
      const auto &target_name = update.target().to_string();
      auto &target = idb[target_name];

      hINFO("Running stratum with target: " << target_name);

      TRACE_START(dataflow_evaluate_preprocess);
      auto active_df =
          ActiveDataflow<Prov>::from_dataflow(update.dataflow(), idb);

      auto dependencies{active_df->dependencies()};

      // the relations which are dependencies of this update
      // and are also a target of any update in this stratum
      std::vector<std::string> changeable_dependencies;
      for (const auto &dep : dependencies) {
        if (std::find(target_relations.begin(), target_relations.end(), dep) !=
            target_relations.end()) {
          changeable_dependencies.push_back(dep);
        }
      }

      // if no dependency of this update can change, then we can skip
      // this update except for on the first iteration.
      if (iteration > 0 && changeable_dependencies.size() == 0) {
        TRACE_END(dataflow_evaluate_preprocess);
        continue;
      }

      // if all dependencies that can change don't currently have any facts
      // in the recent table, then we can skip this update
      if (iteration > 0 && std::all_of(changeable_dependencies.begin(),
                                       changeable_dependencies.end(),
                                       [&idb](const std::string &dep) {
                                         auto table = idb[dep].recent();
                                         return table.size() == 0;
                                       })) {
        TRACE_END(dataflow_evaluate_preprocess);
        continue;
      }
      TRACE_END(dataflow_evaluate_preprocess);

      hINFO("Dataflow: " << *active_df);

      TRACE_START(dataflow_evaluate_process);
      auto output = active_df->evaluate_recent(idb, ctx);

      // Clone the output if the active dataflow is just a source so that the
      // mutations caused by incorporate delta don't mess with the database
      if (dynamic_cast<ActiveDataflowSource<Prov> *>(active_df.get()) !=
          nullptr) {
        output = output.clone();
      }

      output.validate();
      TRACE_END(dataflow_evaluate_process);

      TRACE_START(dataflow_evaluate_postprocess);

      target.delta().append(output);
      TRACE_END(dataflow_evaluate_postprocess);
    }

    if (LOG_STRATUM) {
      std::cout << "IDB after updates in iteration " << iteration << std::endl
                << idb << std::endl;
    }

    DeviceAlloc::singleton_alloc().new_leader();

    auto changed = idb.incorporate_delta(ctx, DeviceAlloc().singleton_alloc());

    hINFO(RED << "Changed in iteration " << iteration << ": ");
    for (auto &name : changed) {
      (void)name;
      hINFO(name);
    }
    hINFO(RESET);
    hINFO("IDB after iteration" << iteration);
    hINFO(idb);

    //std::cout << "Step " << iteration << " done" << std::endl;
    //std::cout << "\tAllocs: "
    //          << (float)lobster_global_allocator().used() / 1024.0 / 1024.0 /
    //                 1024.0
    //          << " GB" << std::endl;
    iter_allocs() = 0;

    if (DeviceAlloc::singleton_alloc().get_leapfrog()) {
      std::cout << "\t\tLeapfrog follower allocs: "
                << (float)DeviceAlloc::singleton_alloc().get_leapfrog()->follower_size() / 1024.0 / 1024.0 / 1024.0
                << " GB" << std::endl;
      std::cout << "\t\tLeapfrog leader allocs: "
                << (float)DeviceAlloc::singleton_alloc().get_leapfrog()->leader_size() / 1024.0 / 1024.0 / 1024.0
                << " GB" << std::endl;
    }

    DeviceAlloc::singleton_alloc().forget_follower();
    DeviceAlloc::singleton_alloc().reset();

    iteration += 1;

    if (changed.size() == 0) {
      break;
    }
  }

  main_timer.stop();
  main_timer.print();

  if (LOG_STRATUM) {
    std::cout << "EXECUTED STRATUM" << std::endl;
    std::cout << "IDB after stratum" << std::endl << idb << std::endl;
  }
}

extern "C" void magic_trace_stop_indicator() {}

template <typename Prov>
Array<StaticDB<Prov>> *execute_stratum_set(
    const Array<Stratum<Prov>> *stratum_set,
    const Array<String> *output_relations, const Array<StaticDB<Prov>> *edbs,
    const Array<StaticDB<Prov>> *idbs, Prov &ctx) {
  int *ptr;
  cudaMalloc(&ptr, sizeof(int));

  TRACE_START(execute_stratum_set);

  magic_trace_stop_indicator();

  bool LOG_STRATUM = std::getenv("LOG_STRATUM") != nullptr;

  TRACE_START(free_last_ret_value);
  if (last_ret_value != nullptr) {
    if (TRACK_ALLOC) {
      std::cout << "Num Managed allocations before free: "
                << ManagedAlloc::allocations().size() << std::endl;
      std::cout << "Num Device allocations before free: "
                << DeviceAlloc::allocations().size() << std::endl;
    }
    delete static_cast<Array<StaticDB<Prov>> *>(last_ret_value);
    if (TRACK_ALLOC) {
      std::cout << "Num Managed allocations after free: "
                << ManagedAlloc::allocations().size() << std::endl;
      std::cout << "Num Device allocations before free: "
                << DeviceAlloc::allocations().size() << std::endl;
    }
  }
  TRACE_END(free_last_ret_value);

  auto batch_size = edbs->size();
  set_batch_size(batch_size);

  if (SAFETY) {
    if (batch_size != idbs->size()) {
      throw std::runtime_error("EDB and IDB batch sizes do not match");
    }
  }

  if (LOG_STRATUM) {
    std::cout << "Stratum set details:" << std::endl;
    std::cout << "edbs: " << *edbs << std::endl;
    std::cout << "idbs: " << *idbs << std::endl;
  }

  try {
    std::unordered_map<std::string, DynamicRelation<Prov>> all_relations;
    auto import_batched_relation = [&all_relations](
                                       std::string &name, size_t index,
                                       const Array<StaticDB<Prov>> &dbs) {
      size_t total_rows = 0;
      for (size_t sample = 0; sample < dbs.size(); sample++) {
        total_rows += dbs[sample].relations()[index].tuples()[0].size();
      }
      TupleType schema(dbs[0].relations()[index].schema());
      auto width = schema.width();
      Array<Array<Value>> facts(width);
      Array<typename Prov::Tag> tags;
      std::vector<char> sample_mask;

      for (size_t col = 0; col < width; col++) {
        new (&facts[col]) Array<Value>(total_rows);
      }

      auto start = 0;
      for (size_t sample = 0; sample < dbs.size(); sample++) {
        const Array<Array<Value>> &ffi_facts =
            dbs[sample].relations()[index].tuples();
        auto sample_size = ffi_facts[0].size();

        // copy facts
        for (size_t col = 0; col < width; col++) {
          std::copy(ffi_facts[col].data(), ffi_facts[col].data() + sample_size,
                    facts[col].data() + start);
        }

        // copy tags
        if (!Prov::is_unit) {
          tags = Array<typename Prov::Tag>(total_rows);
          const Array<typename Prov::FFITag> &foreign_tags =
              dbs[sample].relations()[index].tags();
          for (size_t row = 0; row < sample_size; row++) {
            new (&tags[start + row])
                typename Prov::Tag(Prov::from_foreign(foreign_tags[row]));
          }
        }

        // build sample mask
        std::fill_n(std::back_inserter(sample_mask), sample_size, sample);

        start += sample_size;
      }

      Table<Prov> table(schema, tags, facts, sample_mask);
      table.validate();

      DynamicRelation<Prov> relation(schema);
      relation.stable() = table;

      relation.validate();

      all_relations.insert(std::make_pair(name, std::move(relation)));
    };

    TRACE_START(add_to_all_relations);
    const auto &first_edb = (*edbs)[0];
    for (size_t i = 0; i < first_edb.relations().size(); i++) {
      std::string name = first_edb.relations()[i].predicate().to_string();
      import_batched_relation(name, i, *edbs);
    }
    const auto &first_idb = (*idbs)[0];
    for (size_t i = 0; i < first_idb.relations().size(); i++) {
      std::string name = first_idb.relations()[i].predicate().to_string();
      import_batched_relation(name, i, *idbs);
    }
    for (size_t i = 0; i < stratum_set->size(); i++) {
      const auto &stratum = (*stratum_set)[i];
      for (size_t j = 0; j < stratum.relations().size(); j++) {
        const auto &relation = stratum.relations()[j];
        all_relations.insert(
            std::make_pair(relation.predicate().to_string(),
                           DynamicRelation<Prov>(relation.schema())));
      }
    }
    TRACE_END(add_to_all_relations);

    TRACE_START(construct_idb);
    DynamicIdb idb(std::move(all_relations));
    TRACE_END(construct_idb);

    hINFO("Constructed DynamicIDB:");
    hINFO(idb);

    idb.validate();

    // run stratums
    for (size_t i = 0; i < stratum_set->size(); i++) {
      execute_stratum_device((*stratum_set)[i], idb, ctx);
    }

    idb.validate();

    hINFO("IDB after all stratums:");
    hINFO(idb);

    TRACE_START(construct_output_idb);
    Array<StaticDB<Prov>> new_idbs(batch_size);

    // for each sample in the batch, the set of relations in the output DB
    hINFO("Constructing output IDB relations");
    Array<Array<Relation<Prov>>> new_idb_relations(batch_size);
    for (size_t sample = 0; sample < batch_size; sample++) {
      new (&new_idb_relations[sample])
          Array<Relation<Prov>>(output_relations->size());
    }

    for (size_t i = 0; i < output_relations->size(); i++) {
      auto name = (*output_relations)[i];
      hINFO("Constructing relation " << name.to_string());
      const auto &rel = idb[name.to_string()];
      const auto &table = rel.stable();
      table.validate();

      auto sample_sizes = table.sample_sizes();

      hINFO("Move tags to host")
      // move tags to host
      std::vector<typename Prov::Tag> host_tags = table.tags().to_host();
      hINFO("Move facts to host")
      // move facts to host
      Array<Array<Value>> host_facts(table.width());
      for (size_t col = 0; col < table.width(); col++) {
        Value::Tag value_kind = Value::tag_from_type(
            table.schema().flatten().at(col).singleton().tag());
        new (&host_facts[col]) Array<Value>;
        host_facts[col] = table.column_buffer(col).to_host_tagged(value_kind);
      }
      auto foreign_schema = table.schema().to_foreign();

      hINFO("loop")
      auto start = 0;
      for (size_t sample = 0; sample < batch_size; sample++) {
        auto sample_size = sample_sizes.at(sample);
        // convert tags
      hINFO("convert tags, sample size: " << sample_size)
        Array<typename Prov::FFITag> foreign_tags(sample_size);
        //if (!Prov::is_unit) {
        //  for (size_t j = start; j < start + sample_size; j++) {
        //    hINFO("convert tag " << j - start << " of " << sample_size);
        //    foreign_tags[j - start] = Prov::to_foreign(host_tags[j]);
        //  }
        //}

        // convert facts
      hINFO("convert facts")
        Array<Array<Value>> facts(table.width());
        for (size_t col = 0; col < table.width(); col++) {
          new (&facts[col]) Array<Value>(sample_size);
          for (size_t row = 0; row < sample_size; row++) {
            facts[col][row] = host_facts[col][row + start];
          }
        }
        hINFO("placement new")
        new (&new_idb_relations[sample][i])
            Relation<Prov>(name, foreign_schema, sample_size,
                           std::move(foreign_tags), std::move(facts));
        start += sample_size;
      }
    }

    for (size_t sample = 0; sample < batch_size; sample++) {
      new (&new_idbs[sample])
          StaticDB<Prov>(std::move(new_idb_relations[sample]));
    }

    if (LOG_STRATUM) {
      std::cout << "Constructed new_idbs:" << std::endl;
      std::cout << new_idbs << std::endl;
    }

    Array<StaticDB<Prov>> *return_ptr =
        new Array<StaticDB<Prov>>(std::move(new_idbs));
    TRACE_END(construct_output_idb);

    last_ret_value = return_ptr;
    return return_ptr;
  } catch (std::exception &e) {
    std::cout << "libsclgpu threw an exception: " << e.what() << std::endl;
    return nullptr;
  }
}

__global__ void dummy_kernel() { return; }
extern "C" void libsclgpu_init() {
  static bool libsclgpu_initialized = false;
  if (libsclgpu_initialized) {
    return;
  }
  libsclgpu_initialized = true;

  cudaFree(0);
  dummy_kernel<<<1, 1>>>();
  cudaCheck(cudaDeviceSynchronize());

  cudaMemPool_t mempool;
  cudaDeviceGetDefaultMemPool(&mempool, 0);
  uint64_t threshold = UINT64_MAX;
  cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold);

  lobster_global_allocator() = DeviceAlloc::singleton_alloc();
}

extern "C" void *execute_stratum_device_raw(void *provenance, void *stratum_set,
                                            void *output_relations, void *edbs,
                                            void *idbs) {
  TRACE_START(execute_stratum_device_raw);
  // initialize CUDA context
  TRACE_START(CUDA_init);
  cudaFree(0);
  TRACE_END(CUDA_init);

  DeviceAlloc::init();

  auto log_post_execution_data = []() {
    print_all_tables();

    if constexpr (TRACK_ALLOC) {
      int total_size{0};
      for (auto &[ptr, data] : HostAlloc::allocations()) {
        total_size += std::get<0>(data);
      }
      std::cout << "HostAlloc:\n";
      std::cout << "Unfreed allocations: " << HostAlloc::allocations().size()
                << "\n";
      std::cout << "Total size: " << total_size << std::endl;

      total_size = 0;
      for (auto &[ptr, data] : DeviceAlloc::allocations()) {
        total_size += std::get<1>(data);
      }
      std::cout << "DeviceAlloc:\n";
      std::cout << "Unfreed allocations: " << DeviceAlloc::allocations().size()
                << "\n";
      std::cout << "Total size: " << total_size << std::endl;

      std::cout << "ManagedAlloc:\n";
      std::cout << "Unfreed allocations: " << ManagedAlloc::allocations().size()
                << "\n";
      std::cout << "All allocations: " << ManagedAlloc::all_allocations().size()
                << std::endl;
      // managed alloc
      {
        std::ofstream managed_alloc_file("managed_allocs.txt");
        for (auto &[ptr, pair] : ManagedAlloc::allocations()) {
          auto [size, source] = pair;
          managed_alloc_file << ptr << " " << size << " " << source
                             << std::endl;
        }
        managed_alloc_file.close();
      }
      {
        std::ofstream managed_alloc_file("managed_allocs_all.txt");
        for (auto &[ptr, size] : ManagedAlloc::all_allocations()) {
          managed_alloc_file << ptr << " " << size << " " << std::endl;
        }
        managed_alloc_file.close();
      }
      // device alloc
      {
        std::ofstream device_alloc_file("device_allocs.txt");
        for (auto &[ptr, pair] : DeviceAlloc::allocations()) {
          auto [index, size, source] = pair;
          device_alloc_file << ptr << " " << index << " " << size << " "
                            << source << std::endl;
        }
        device_alloc_file.close();
      }
      // host alloc
      {
        std::ofstream host_alloc_file("host_allocs.txt");
        for (auto &[ptr, pair] : HostAlloc::allocations()) {
          auto [size, source] = pair;
          host_alloc_file << ptr << " " << size << " " << source << std::endl;
        }
        host_alloc_file.close();
      }
    }
  };

  Provenance *prov = reinterpret_cast<Provenance *>(provenance);

  if (prov->tag == Provenance::Tag::DiffTopKProofs) {
    std::cout << "Input facts sizes: ";
    for (size_t i = 0; i < prov->diff_top_k_proofs.literal_probabilities.size();
         i++) {
      std::cout << prov->diff_top_k_proofs.literal_probabilities[i].size()
                << " ";
    }
    std::cout << std::endl;
  }

  TRACE_START(prov_init);
  void *output;
  DISPATCH_ON_PROV(
      prov, Prov, ctx,
      TRACE_END(prov_init);
      output = execute_stratum_set<Prov>((Array<Stratum<Prov>> *)stratum_set,
                                         (Array<String> *)output_relations,
                                         (Array<StaticDB<Prov>> *)edbs,
                                         (Array<StaticDB<Prov>> *)idbs, ctx););

  if (TRACK_ALLOC) {
    std::cout << "Max allocation size on device: " << DeviceAlloc::max_alloc()
              << " (GB: " << DeviceAlloc::max_alloc() / 1024.0 / 1024.0 / 1024.0
              << ")" << std::endl;
  }

  // reset arena allocator (if in use)
  // DeviceAlloc::reset()

  log_post_execution_data();
  return output;
}
