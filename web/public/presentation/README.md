# Turbofan RUL presentation

This 20-slide browser presentation explains the C-MAPSS data, model selection, results, interpretability, reliability, and application boundaries. Its figures and numbers use the committed research snapshot described in [the results document](../../../docs/RESULTS.md). The sensor trace samples FD001 training engine 1 every four cycles, with the final cycle also included.

From the repository root, serve the existing web assets:

```powershell
python -m http.server 8000 --directory web/public
```

Open `http://localhost:8000/presentation/`. Use the arrow keys, Page Up, Page Down, Home, End, or on-screen buttons to navigate. Fullscreen and browser print-to-PDF controls are available in the toolbar. Print uses a 16:9 page for each slide; enable background graphics in the browser's print settings.

The web deck is the source for a later PowerPoint conversion. Print-to-PDF produces a PDF, not an editable PowerPoint file. Live sensor streaming, SHAP explanations, model pruning, and quantization are future work, not implemented claims.
