#!/bin/bash

echo "🧪 Testing Gemma Rust Application"
echo "=================================="

echo ""
echo "🔨 Building the application..."
cargo build

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
else
    echo "❌ Build failed!"
    exit 1
fi

echo ""
echo "🚀 Running the application..."
echo "Type 'quit' to exit the test"
echo ""

# Run the application
cargo run

echo ""
echo "🏁 Test completed!"
