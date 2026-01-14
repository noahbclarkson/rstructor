/// Container-level attributes for structs and enums
#[derive(Debug, Clone)]
pub struct ContainerAttributes {
    /// Description of the struct or enum
    pub description: Option<String>,

    /// Custom title for the schema (overrides the default type name)
    pub title: Option<String>,

    /// Examples of valid instances (as tokenstreams)
    pub examples: Vec<proc_macro2::TokenStream>,

    /// Serde rename_all case style (from serde attribute)
    pub serde_rename_all: Option<String>,

    /// Serde tag for internally/adjacently tagged enums
    pub serde_tag: Option<String>,

    /// Serde content field name for adjacently tagged enums
    pub serde_content: Option<String>,

    /// Serde untagged marker for enums
    pub serde_untagged: bool,

    /// Custom validation function path (e.g., "validate_product" or "my_module::validate")
    pub validate: Option<String>,
}

impl ContainerAttributes {
    /// Create a new container attributes object
    pub fn new(
        description: Option<String>,
        title: Option<String>,
        examples: Vec<proc_macro2::TokenStream>,
        serde_rename_all: Option<String>,
        serde_tag: Option<String>,
        serde_content: Option<String>,
        serde_untagged: bool,
        validate: Option<String>,
    ) -> Self {
        Self {
            description,
            title,
            examples,
            serde_rename_all,
            serde_tag,
            serde_content,
            serde_untagged,
            validate,
        }
    }

    /// Returns true if there are no attributes set - currently unused but kept for future use
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.description.is_none()
            && self.title.is_none()
            && self.examples.is_empty()
            && self.serde_rename_all.is_none()
            && self.serde_tag.is_none()
            && self.serde_content.is_none()
            && !self.serde_untagged
            && self.validate.is_none()
    }
}
