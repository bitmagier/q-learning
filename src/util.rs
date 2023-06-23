use log::LevelFilter;

pub fn init_logging() {
    env_logger::builder()
        .format_target(false)
        .format_timestamp_secs()
        .filter_level(LevelFilter::Info)
        .init()
}
