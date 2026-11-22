# RUL method anatomy

`rul-workbench/build.mjs` generates the static SVG used on slide 8. It diagrams the method stages: retained sensor channels, a 30-cycle input window, Ridge and LSTM model paths, and validation selection. It is an explanatory schematic, not a physical C-MAPSS engine model.

The projection helper in `vendor/iso-kit.mjs` is distributed under the MIT License. Its copyright and permission notice are retained in `vendor/LICENSE`.

Regenerate the figure from the repository root with:

```powershell
node docs/visualizations/anatomy/rul-workbench/build.mjs
```
