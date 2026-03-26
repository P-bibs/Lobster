pub use flame::dump_html;

// The below code is copied from the crate flamescope
// (https://docs.rs/flamescope/latest/flamescope/). The source needs to be compiled as part of
// scallop-core and not as a dependency so that the global state that flamescope uses is accessible
// by libsclgpu.

// ======================= flamescope ===============================

pub mod flamescope {

  use flame::Span;
  use serde::{Deserialize, Serialize};
  use std::borrow::Cow;
  use std::cell::{Cell, RefCell};
  use std::io::{Error as IoError, Write};
  use std::iter::Peekable;
  use std::sync::Mutex;
  use std::time::{Duration, Instant};
  pub type StrCow = Cow<'static, str>;

  #[derive(Debug, PartialEq, Serialize, Deserialize)]
  #[serde(rename_all = "camelCase")]
  pub struct SpeedscopeFile {
    #[serde(rename = "$schema")]
    pub schema: &'static str,

    pub profiles: Vec<Profile>,
    pub shared: Shared,

    pub active_profile_index: Option<u64>,

    pub exporter: Option<String>,

    pub name: Option<String>,
  }

  #[derive(Debug, PartialEq, Serialize, Deserialize)]
  #[serde(tag = "type")]
  #[serde(rename_all = "camelCase")]
  pub enum Profile {
    #[serde(rename_all = "camelCase")]
    Sampled {
      name: StrCow,
      unit: ValueUnit,
      start_value: u64,
      end_value: u64,
      samples: Vec<SampledStack>,
      weights: Vec<u64>,
    },
    #[serde(rename_all = "camelCase")]
    Evented {
      name: StrCow,
      unit: ValueUnit,
      start_value: u64,
      end_value: u64,
      events: Vec<Event>,
    },
  }

  #[derive(Debug, PartialEq, Serialize, Deserialize)]
  pub struct Event {
    #[serde(rename = "type")]
    pub event_type: EventType,
    pub at: u64,
    pub frame: usize,
  }

  #[derive(Debug, PartialEq, Serialize, Deserialize)]
  pub enum EventType {
    #[serde(rename = "O")]
    OpenFrame,
    #[serde(rename = "C")]
    CloseFrame,
  }

  type SampledStack = Vec<usize>;

  #[derive(Debug, PartialEq, Serialize, Deserialize)]
  pub struct Shared {
    pub frames: Vec<Frame>,
  }

  #[derive(Debug, PartialEq, Clone, Eq, Hash, Serialize, Deserialize)]
  pub struct Frame {
    pub name: StrCow,
    pub file: Option<String>,
    pub line: Option<u32>,
    pub col: Option<u32>,
  }

  impl Frame {
    #[inline]
    pub fn new(name: StrCow) -> Frame {
      Frame {
        name,
        file: None,
        line: None,
        col: None,
      }
    }
  }

  #[derive(Debug, PartialEq, Serialize, Deserialize)]
  #[serde(rename_all = "lowercase")]
  pub enum ValueUnit {
    Bytes,
    Microseconds,
    Milliseconds,
    Nanoseconds,
    None,
    Seconds,
  }

  use indexmap::IndexSet;

  const JSON_SCHEMA_URL: &str = "https://www.speedscope.app/file-format-schema.json";

  /// Convert flame spans to the speedscope profile format.
  pub fn spans_to_speedscope(spans: Vec<Span>) -> SpeedscopeFile {
    let mut frames = IndexSet::new();
    let profiles = spans
      .into_iter()
      .map(|span| Profile::Evented {
        name: span.name.clone(),
        unit: ValueUnit::Nanoseconds,
        start_value: span.start_ns,
        end_value: span.end_ns,
        events: {
          let mut events = Vec::new();
          span_extend_events(&mut frames, &mut events, span);
          events
        },
      })
      .collect();
    SpeedscopeFile {
      // always the same
      schema: JSON_SCHEMA_URL,
      active_profile_index: None,
      exporter: None,
      name: None,
      profiles,
      shared: Shared {
        frames: frames.into_iter().collect(),
      },
    }
  }

  fn span_extend_events(frames: &mut IndexSet<Frame>, events: &mut Vec<Event>, span: Span) {
    let (frame, _) = frames.insert_full(Frame::new(span.name));
    events.push(Event {
      event_type: EventType::OpenFrame,
      at: span.start_ns,
      frame,
    });
    for child in span.children {
      span_extend_events(frames, events, child);
    }
    events.push(Event {
      event_type: EventType::CloseFrame,
      at: span.end_ns,
      frame,
    });
  }

  #[inline]
  pub fn dump(writer: impl Write) -> serde_json::Result<()> {
    write_spans(writer, flame::spans())
  }

  #[inline]
  pub fn write_spans(writer: impl Write, spans: Vec<Span>) -> serde_json::Result<()> {
    let speedscope = spans_to_speedscope(spans);
    serde_json::to_writer(writer, &speedscope)
  }
}

// ============================ FFI  ================================
use std::ffi::CStr;
use std::fs::File;
use std::os::raw::c_char;

#[no_mangle]
pub extern "C" fn flame_start(name: *const c_char) {
  let result = std::panic::catch_unwind(|| {
    let name = unsafe { CStr::from_ptr(name).to_str().unwrap().to_owned() };
    flame::start(name);
  });
  if result.is_err() {
    eprintln!("error: rust panicked");
  }
}

#[no_mangle]
pub extern "C" fn flame_end(name: *const c_char) {
  let result = std::panic::catch_unwind(|| {
    let name = unsafe { CStr::from_ptr(name).to_str().unwrap().to_owned() };
    flame::end(name);
  });
  if result.is_err() {
    eprintln!("error: rust panicked");
  }
}

#[no_mangle]
pub extern "C" fn flame_dump(path: *const c_char) {
  let path = unsafe { CStr::from_ptr(path).to_str().unwrap() };
  flamescope::dump(&mut File::create(path).unwrap()).unwrap();
}

#[no_mangle]
pub extern "C" fn flame_dump_html(path: *const c_char) {
  let path = unsafe { CStr::from_ptr(path).to_str().unwrap() };
  flame::dump_html(&mut File::create(path).unwrap()).unwrap();
}

#[no_mangle]
pub extern "C" fn flame_debug() {
  flame::debug();
}

#[no_mangle]
pub extern "C" fn flame_dump_stdout() {
  flame::dump_stdout();
}

#[no_mangle]
pub extern "C" fn flame_clear() {
  flame::clear();
}
