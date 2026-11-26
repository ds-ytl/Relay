#!/bin/sh
cd /Users/taoyongwen/Projects/relay || exit 1
exec /Users/taoyongwen/.cargo/bin/cargo run -q -p relay-desktop
