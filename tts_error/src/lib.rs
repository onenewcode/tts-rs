use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    InvalidArgument,
    NotFound,
    Conflict,
    Unsupported,
    Internal,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiagnosticContext {
    pub key: String,
    pub value: String,
}

#[derive(Debug, Error)]
#[error("[{code}] {message}")]
pub struct DiagnosticError {
    category: ErrorCategory,
    code: String,
    message: String,
    context: Vec<DiagnosticContext>,
}

impl DiagnosticError {
    pub fn new(
        category: ErrorCategory,
        code: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            category,
            code: code.into(),
            message: message.into(),
            context: Vec::new(),
        }
    }

    pub fn invalid_argument(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(ErrorCategory::InvalidArgument, code, message)
    }

    pub fn not_found(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(ErrorCategory::NotFound, code, message)
    }

    pub fn conflict(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(ErrorCategory::Conflict, code, message)
    }

    pub fn unsupported(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(ErrorCategory::Unsupported, code, message)
    }

    pub fn internal(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(ErrorCategory::Internal, code, message)
    }

    pub fn with_context(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.context.push(DiagnosticContext {
            key: key.into(),
            value: value.into(),
        });
        self
    }

    pub fn category(&self) -> ErrorCategory {
        self.category
    }

    pub fn code(&self) -> &str {
        &self.code
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn context(&self) -> &[DiagnosticContext] {
        &self.context
    }

    pub fn render(&self) -> RenderedDiagnostic {
        RenderedDiagnostic {
            category: self.category,
            code: self.code.clone(),
            message: self.message.clone(),
            context: self.context.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderedDiagnostic {
    pub category: ErrorCategory,
    pub code: String,
    pub message: String,
    pub context: Vec<DiagnosticContext>,
}
