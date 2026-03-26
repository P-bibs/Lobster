use std::ffi::CString;

#[derive(Debug, Clone)]
#[repr(C)]
pub struct C_Array<T> {
  pub length: usize,
  pub values: *mut T,
}

unsafe impl<T> Send for C_Array<T> {}

impl<T: std::fmt::Display> std::fmt::Display for C_Array<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "[")?;
    for i in 0..self.length {
      write!(f, "{}", unsafe { &*self.values.add(i) })?;
      if i < self.length - 1 {
        write!(f, ", ")?;
      }
    }
    write!(f, "]")
  }
}

impl<T> C_Array<T> {
  pub fn new(values: Vec<T>) -> Self {
    let slice = values.into_boxed_slice();
    let length = slice.len();
    let values = Box::into_raw(slice) as *mut T;
    Self { length, values }
  }

  pub fn length(&self) -> usize {
    self.length
  }

  pub fn empty() -> Self {
    Self {
      length: 0,
      values: std::ptr::null_mut(),
    }
  }

  pub fn as_slice(&self) -> &[T] {
    unsafe {
      if self.length == 0 {
        std::slice::from_raw_parts(std::ptr::NonNull::dangling().as_ptr(), 0)
      } else {
        std::slice::from_raw_parts(self.values, self.length)
      }
    }
  }

  pub fn into_vec(self) -> Vec<T> {
    unsafe {
      if self.length == 0 {
        Vec::new()
      } else {
        Vec::from_raw_parts(self.values, self.length, self.length)
      }
    }
  }

  pub fn get(&self, index: usize) -> Option<&T> {
    if index < self.length {
      Some(unsafe { &*self.values.add(index) })
    } else {
      None
    }
  }
}

impl<T: Clone> C_Array<T> {
  pub fn clone_to_vec(&self) -> Vec<T> {
    let slice = self.as_slice();
    slice.to_vec()
  }
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct C_String {
  pub length: usize,
  pub value: *mut u8,
}
unsafe impl Send for C_String {}
unsafe impl Sync for C_String {}
impl std::fmt::Display for C_String {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "\"")?;
    for i in 0..self.length {
      write!(f, "{}", unsafe { char::from(*self.value.add(i)) })?;
    }
    write!(f, "\"")
  }
}
impl C_String {
  pub fn new(value: String) -> Self {
    let length = value.len();
    let value = CString::new(value).unwrap().into_raw() as *mut u8;
    Self { length, value }
  }
  pub fn to_string(&self) -> String {
    let mut result = String::new();
    for i in 0..self.length {
      result.push(unsafe { char::from(*self.value.add(i)) });
    }
    result
  }
}

impl std::cmp::PartialEq for C_String {
  fn eq(&self, other: &Self) -> bool {
    if self.length != other.length {
      return false;
    }
    for i in 0..self.length {
      if unsafe { *self.value.add(i) } != unsafe { *other.value.add(i) } {
        return false;
      }
    }
    true
  }
}
