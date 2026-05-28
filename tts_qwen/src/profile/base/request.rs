use crate::profile::QwenRequest;

#[derive(Debug, Clone)]
pub(crate) struct BaseRequest {
    pub(crate) text: String,
    pub(crate) language: Option<String>,
    source: QwenRequest,
}

impl BaseRequest {
    pub(crate) fn source(&self) -> &QwenRequest {
        &self.source
    }
}

impl From<QwenRequest> for BaseRequest {
    fn from(request: QwenRequest) -> Self {
        Self {
            text: request.text.clone(),
            language: request.language.clone(),
            source: request,
        }
    }
}
