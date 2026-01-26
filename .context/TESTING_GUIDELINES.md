1. Use `SyntheticDataGenerator` to create synthetic data fixtures to be leveraged across multiple module testing where applicable.
2. IMPORTANT: Keep tests at a **minimum** to reduce verbosity & redundancy and improve maintainability, while ensuring adequate coverage!! This is even more true as we are in early stages of development.
3. Ensure that all tests are idempotent, meaning they can be run multiple times without changing the result beyond the initial application.
4. Follow best practices for writing clean, readable, and maintainable test code.
5. Keep tests mostly unit oriented. Integration tests across modules will be implemented once the modules are more mature. Integration within modules should be tested however (e.g., `SyntheticDataGenerator` integration)
6. Use descriptive names for test functions to clearly indicate their purpose.
