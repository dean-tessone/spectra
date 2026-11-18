"""Re-render every figure (Python/matplotlib, nature-figure skill)."""
import fig1, fig2, fig3, fig4, fig5, fig6, figED1, figS1, figS2, figS3, figS4, figS7, figS8, figS9_sites   # fig5 = molecular; figS9_sites = Supplementary Fig. 3 (revision 2026-11-17)
for m in (fig1, fig2, fig3, fig4, fig5, fig6, figED1, figS2, figS9_sites, figS4, figS3, figS7, figS8):
    m.build()
