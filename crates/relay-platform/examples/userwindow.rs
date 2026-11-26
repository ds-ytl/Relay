fn main() {
    println!("front_window: {:?}", relay_platform::front_window());
    println!(
        "user_front_window: {:?}",
        relay_platform::user_front_window()
    );
    println!("is_shell(relay): {}", relay_platform::is_shell_app("relay"));
}
