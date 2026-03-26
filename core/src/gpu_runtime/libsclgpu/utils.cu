#include "utils.h"

int batch_size = -1;
int get_batch_size() {
  return batch_size;
}
void set_batch_size(int size) {
  batch_size = size;
}
