"""
Custom Exception Classes
========================

Custom exceptions for the Bank Churn Prediction system.
Provides specific error types for different failure modes to enable better error handling.
"""

from typing import Any, Dict, Optional
import traceback


class BankChurnException(Exception):
    """Base exception class for all bank churn prediction errors"""
    
    def __init__(
        self, 
        message: str, 
        details: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None
    ):
        self.message = message
        self.details = details or {}
        self.error_code = error_code
        self.traceback = traceback.format_exc()
        super().__init__(self.message)
    
    def __str__(self) -> str:
        base_message = f"{self.__class__.__name__}: {self.message}"
        
        if self.error_code:
            base_message = f"[{self.error_code}] {base_message}"
            
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            base_message = f"{base_message} | Details: {details_str}"
            
        return base_message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON serialization"""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "traceback": self.traceback
        }


class DataValidationError(BankChurnException):
    """
    Raised when data validation fails.
    
    Examples:
    - Missing required columns
    - Invalid data types
    - Out-of-range values
    - Corrupted data files
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, "E001")


class DataProcessingError(BankChurnException):
    """
    Raised when data processing operations fail.
    
    Examples:
    - Failed data transformations
    - Encoding errors
    - Scaling failures
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, "E002")


class FeatureEngineeringError(BankChurnException):
    """
    Raised when feature engineering fails.
    
    Examples:
    - Feature creation errors
    - Invalid feature transformations
    - Missing feature dependencies
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, "E003")


class ModelTrainingError(BankChurnException):
    """
    Raised when model training fails.
    
    Examples:
    - Convergence failures
    - Invalid hyperparameters
    - Insufficient training data
    - Out of memory errors
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, "E004")


class ModelLoadError(BankChurnException):
    """
    Raised when loading a trained model fails.
    
    Examples:
    - Model file not found
    - Corrupted model file
    - Version mismatch
    - Missing model artifacts
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, "E005")


class InferenceError(BankChurnException):
    """
    Raised when prediction/inference fails.
    
    Examples:
    - Feature mismatch
    - Invalid input data
    - Model prediction errors
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, "E006")


class ConfigurationError(BankChurnException):
    """
    Raised when configuration is invalid or missing.
    
    Examples:
    - Missing config files
    - Invalid config format
    - Missing required parameters
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, "E007")


class FileOperationError(BankChurnException):
    """
    Raised when file operations fail.
    
    Examples:
    - File not found
    - Permission denied
    - Disk space issues
    - Invalid file format
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, "E008")


class SHAPExplanationError(BankChurnException):
    """
    Raised when SHAP explanation computation fails.
    
    Examples:
    - SHAP computation errors
    - Incompatible model types
    - Memory issues with SHAP
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, "E009")


class DriftDetectionError(BankChurnException):
    """
    Raised when data drift detection fails.
    
    Examples:
    - Missing baseline statistics
    - Invalid drift thresholds
    - Incompatible data distributions
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, "E010")


class FairnessCheckError(BankChurnException):
    """
    Raised when fairness validation fails.
    
    Examples:
    - Missing demographic data
    - Invalid fairness metrics
    - Bias detection errors
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, "E011")


class ExportError(BankChurnException):
    """
    Raised when data export operations fail.
    
    Examples:
    - Export format errors
    - Missing export directory
    - Permission issues
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, "E012")


class DatabaseError(BankChurnException):
    """
    Raised when database operations fail.
    
    Examples:
    - Connection errors
    - Query execution errors
    - Timeout errors
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, "E013")


class AuthenticationError(BankChurnException):
    """
    Raised when authentication/authorization fails.
    
    Examples:
    - Invalid API keys
    - Permission denied
    - Token expiration
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, "E014")


# === NEW EXCEPTIONS FOR PHASE 7 ===

class ModelLoadingError(BankChurnException):
    """
    Raised when model loading fails (alias for ModelLoadError for compatibility).
    
    Examples:
    - Model file corruption
    - Version incompatibility
    - Missing dependencies
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, "E005")  # Same code as ModelLoadError


class DataValidationError(BankChurnException):
    """
    Raised when data validation fails (already exists, keeping for compatibility).
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, "E001")


# Error code mapping for structured error handling
ERROR_CODES = {
    DataValidationError: "E001",
    DataProcessingError: "E002", 
    FeatureEngineeringError: "E003",
    ModelTrainingError: "E004",
    ModelLoadError: "E005",
    ModelLoadingError: "E005",  # Alias for compatibility
    InferenceError: "E006",
    ConfigurationError: "E007",
    FileOperationError: "E008",
    SHAPExplanationError: "E009",
    DriftDetectionError: "E010",
    FairnessCheckError: "E011",
    ExportError: "E012",
    DatabaseError: "E013",
    AuthenticationError: "E014"
}


def get_error_code(exception: Exception) -> str:
    """Get error code for a given exception type"""
    return ERROR_CODES.get(type(exception), "E999")


def format_error_message(exception: Exception) -> str:
    """Format exception with error code for logging"""
    error_code = get_error_code(exception)
    return f"[{error_code}] {str(exception)}"


def create_exception_from_dict(error_dict: Dict[str, Any]) -> BankChurnException:
    """Create exception instance from dictionary"""
    error_type = error_dict.get("error_type", "BankChurnException")
    message = error_dict.get("message", "Unknown error")
    details = error_dict.get("details", {})
    error_code = error_dict.get("error_code")
    
    # Map error type string to class
    exception_class = globals().get(error_type, BankChurnException)
    
    # Create exception instance
    exception = exception_class(message, details)
    if error_code:
        exception.error_code = error_code
        
    return exception


# Compatibility imports for Phase 7
# These allow the Phase 7 files to use your existing exception classes
ModelLoadingError = ModelLoadError  # Alias for compatibility
DataValidationError = DataValidationError  # Already exists
DriftDetectionError = DriftDetectionError  # Already exists