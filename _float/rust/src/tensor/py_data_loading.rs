//! Python bindings for SRT Data Loading
//!
//! Exposes native CSV/binary parsing to Python without NumPy/Pandas.

use super::data_loading::{
    apply_srt_transforms, golden_normalize, DataBatch, DataLoadError, DataType,
    GoldenExactConverter, SRTBinaryLoader, SRTCSVParser,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Convert DataLoadError to PyErr
fn to_py_err(e: DataLoadError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Python wrapper for DataType enum
#[pyclass]
#[derive(Clone)]
pub enum PyDataType {
    Float32,
    Float64,
    Int32,
    Int64,
    String,
    Bool,
}

impl From<DataType> for PyDataType {
    fn from(dt: DataType) -> Self {
        match dt {
            DataType::Float32 => PyDataType::Float32,
            DataType::Float64 => PyDataType::Float64,
            DataType::Int32 => PyDataType::Int32,
            DataType::Int64 => PyDataType::Int64,
            DataType::String => PyDataType::String,
            DataType::Bool => PyDataType::Bool,
        }
    }
}

impl From<PyDataType> for DataType {
    fn from(dt: PyDataType) -> Self {
        match dt {
            PyDataType::Float32 => DataType::Float32,
            PyDataType::Float64 => DataType::Float64,
            PyDataType::Int32 => DataType::Int32,
            PyDataType::Int64 => DataType::Int64,
            PyDataType::String => DataType::String,
            PyDataType::Bool => DataType::Bool,
        }
    }
}

/// Python wrapper for DataBatch
#[pyclass]
#[derive(Clone)]
pub struct PyDataBatch {
    #[pyo3(get)]
    pub data: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub column_names: Vec<String>,
    #[pyo3(get)]
    pub batch_size: usize,
}

#[pymethods]
impl PyDataBatch {
    /// Get number of rows
    #[getter]
    fn num_rows(&self) -> usize {
        self.data.len()
    }

    /// Get number of columns
    #[getter]
    fn num_cols(&self) -> usize {
        if self.data.is_empty() {
            0
        } else {
            self.data[0].len()
        }
    }

    /// Get a specific row
    fn get_row(&self, idx: usize) -> PyResult<Vec<f64>> {
        self.data
            .get(idx)
            .cloned()
            .ok_or_else(|| PyValueError::new_err(format!("Row {} out of bounds", idx)))
    }

    /// Get a specific column
    fn get_column(&self, idx: usize) -> PyResult<Vec<f64>> {
        if self.data.is_empty() {
            return Ok(vec![]);
        }
        if idx >= self.data[0].len() {
            return Err(PyValueError::new_err(format!(
                "Column {} out of bounds",
                idx
            )));
        }
        Ok(self.data.iter().map(|row| row[idx]).collect())
    }

    /// Flatten to 1D array (row-major)
    fn flatten(&self) -> Vec<f64> {
        self.data.iter().flatten().copied().collect()
    }

    /// Get shape as (rows, cols)
    fn shape(&self) -> (usize, usize) {
        (self.num_rows(), self.num_cols())
    }
}

impl From<DataBatch> for PyDataBatch {
    fn from(batch: DataBatch) -> Self {
        PyDataBatch {
            data: batch.data,
            column_names: batch.column_names,
            batch_size: batch.batch_size,
        }
    }
}

/// Python wrapper for SRTCSVParser
#[pyclass]
pub struct PySRTCSVParser {
    inner: SRTCSVParser,
}

#[pymethods]
impl PySRTCSVParser {
    /// Create a new CSV parser
    #[new]
    #[pyo3(signature = (delimiter=',', has_header=true, batch_size=10000))]
    fn new(delimiter: char, has_header: bool, batch_size: usize) -> Self {
        let parser = SRTCSVParser::new()
            .with_delimiter(delimiter)
            .with_header(has_header)
            .with_batch_size(batch_size);
        PySRTCSVParser { inner: parser }
    }

    /// Parse complete CSV file
    fn parse(&self, path: &str) -> PyResult<PyDataBatch> {
        self.inner
            .parse_complete(path)
            .map(PyDataBatch::from)
            .map_err(to_py_err)
    }

    /// Parse CSV file in streaming batches
    fn parse_streaming(&self, path: &str) -> PyResult<PyStreamingCSVIterator> {
        self.inner
            .parse_streaming(path)
            .map(|iter| PyStreamingCSVIterator { inner: iter })
            .map_err(to_py_err)
    }
}

/// Python wrapper for streaming CSV iterator
#[pyclass]
pub struct PyStreamingCSVIterator {
    inner: super::data_loading::StreamingCSVIterator,
}

#[pymethods]
impl PyStreamingCSVIterator {
    /// Get next batch
    fn next_batch(&mut self) -> PyResult<Option<PyDataBatch>> {
        self.inner
            .next_batch()
            .map(|opt| opt.map(PyDataBatch::from))
            .map_err(to_py_err)
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<PyDataBatch>> {
        slf.next_batch()
    }
}

/// Python wrapper for binary data loader
#[pyclass]
pub struct PySRTBinaryLoader {
    record_size: usize,
    data_types: Vec<DataType>,
}

#[pymethods]
impl PySRTBinaryLoader {
    /// Create a new binary loader
    #[new]
    fn new(record_size: usize, data_types: Vec<PyDataType>) -> Self {
        let types: Vec<DataType> = data_types.into_iter().map(DataType::from).collect();
        PySRTBinaryLoader {
            record_size,
            data_types: types,
        }
    }

    /// Load binary file
    #[pyo3(signature = (path, max_records=None))]
    fn load(&self, path: &str, max_records: Option<usize>) -> PyResult<PyDataBatch> {
        let loader = SRTBinaryLoader::new(self.record_size, self.data_types.clone());
        loader
            .load(path, max_records)
            .map(PyDataBatch::from)
            .map_err(to_py_err)
    }
}

/// Python wrapper for GoldenExact converter
#[pyclass]
pub struct PyGoldenExactConverter {
    precision: i64,
}

#[pymethods]
impl PyGoldenExactConverter {
    #[new]
    #[pyo3(signature = (precision=100))]
    fn new(precision: i64) -> Self {
        PyGoldenExactConverter { precision }
    }

    /// Convert data to GoldenExact representation
    fn convert(&self, data: Vec<f64>) -> Vec<f64> {
        // The GoldenExact conversion snaps values to nearest a + b*phi lattice points
        let converter = GoldenExactConverter::new();
        let mut batch = DataBatch {
            data: vec![data],
            column_names: vec![],
            column_types: vec![],
            batch_size: 1,
        };
        converter.convert_to_golden_exact(&mut batch);
        batch.data.into_iter().next().unwrap_or_default()
    }

    /// Convert batch to SRT tensor format
    fn to_srt_tensor(&self, batch: &PyDataBatch) -> Vec<f64> {
        let inner_batch = DataBatch {
            data: batch.data.clone(),
            column_names: batch.column_names.clone(),
            column_types: vec![],
            batch_size: batch.batch_size,
        };
        let converter = GoldenExactConverter::new();
        let mut result = converter.convert_to_srt_tensor(&inner_batch);
        // Use precision field to round result
        let scale = 10_f64.powi(self.precision as i32);
        result = result.iter().map(|v| (v * scale).round() / scale).collect();
        result
    }

    /// Get precision setting
    #[getter]
    fn get_precision(&self) -> i64 {
        self.precision
    }

    /// Set precision setting
    #[setter]
    fn set_precision(&mut self, precision: i64) {
        self.precision = precision.clamp(0, 1000);
    }
}

// === Standalone Functions ===

/// Normalize data using golden ratio scaling
#[pyfunction]
pub fn py_golden_normalize(mut data: Vec<f64>) -> Vec<f64> {
    golden_normalize(&mut data);
    data
}

/// Apply full SRT transform pipeline
#[pyfunction]
pub fn py_apply_srt_transforms(mut data: Vec<f64>) -> Vec<f64> {
    apply_srt_transforms(&mut data);
    data
}

/// Parse CSV file directly (convenience function)
#[pyfunction]
#[pyo3(signature = (path, delimiter=',', has_header=true))]
pub fn py_parse_csv(path: &str, delimiter: char, has_header: bool) -> PyResult<PyDataBatch> {
    let parser = SRTCSVParser::new()
        .with_delimiter(delimiter)
        .with_header(has_header);
    parser
        .parse_complete(path)
        .map(PyDataBatch::from)
        .map_err(to_py_err)
}

/// Load binary file directly (convenience function)
#[pyfunction]
#[pyo3(signature = (path, record_size, max_records=None))]
pub fn py_load_binary(
    path: &str,
    record_size: usize,
    max_records: Option<usize>,
) -> PyResult<PyDataBatch> {
    let loader = SRTBinaryLoader::new(record_size, vec![DataType::Float64]);
    loader
        .load(path, max_records)
        .map(PyDataBatch::from)
        .map_err(to_py_err)
}
