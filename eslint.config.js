// Flat ESLint config (ESLint 9). CommonJS on purpose - the project has no "type":"module"
// and tsconfig emits commonjs, so require()/module.exports is the right form here.
// Scope: src/**/*.ts only. The engine (Python) and renderer (hand-tuned inline JS) are linted
// separately (pyright) or not at all - see README / package.json scripts.
const tseslint = require('typescript-eslint');

module.exports = tseslint.config(
  { ignores: ['dist/**', 'release/**', 'node_modules/**', 'engine/**', 'renderer/**'] },
  ...tseslint.configs.recommended,
  {
    files: ['src/**/*.ts'],
    rules: {
      // Keep it basic and high-signal for Electron main-process code:
      '@typescript-eslint/no-explicit-any': 'off', // ctypes/IPC glue uses any deliberately
      '@typescript-eslint/no-unused-vars': ['warn', { argsIgnorePattern: '^_', varsIgnorePattern: '^_' }],
    },
  },
);
