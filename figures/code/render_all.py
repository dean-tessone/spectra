"""Re-render every figure (Python/matplotlib, nature-figure skill)."""
import fig1, fig2, fig3, fig4, fig5, figED1, figS1, figS2, figS3, figS4, figS6, figS7, figS8   # figS5 (IMC of individual cells) retired 2026-11-17
for m in (fig1, fig2, fig3, fig4, fig5, figED1, figS2, figS3, figS4, figS6, figS7, figS8):
    m.build()
