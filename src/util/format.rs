use num_format::{CustomFormat, Grouping};

pub fn number_format() -> CustomFormat {
    CustomFormat::builder()
        .grouping(Grouping::Standard)
        .minus_sign("-")
        .separator("_")
        .build()
        .unwrap()
}