use log::LevelFilter;

pub fn init_logging() {
    env_logger::builder()
        .format_target(false)
        .format_timestamp_secs()
        .filter_level(LevelFilter::Info)
        //.parse_default_env()
        .init()
}

#[cfg(test)]
#[ctor::ctor]
fn init() {
    use log::LevelFilter;
    env_logger::builder()
        // .format_target(false)
        .format_timestamp_secs()
        .filter_level(LevelFilter::Debug)
        .parse_default_env()
        .init()
}
