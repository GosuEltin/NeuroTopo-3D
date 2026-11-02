"""Build Official JCDE Manuscript Edition, Compile PDF, and Package All 8 Submission Deliverables.

Creates:
  1. `jcde_submission_package/manuscript_jcde/` (complete JCDE-formatted LaTeX project)
  2. `jcde_submission_package/1_Main_Document_JCDE.pdf` (compiled single-blind JCDE review PDF)
  3. `jcde_submission_package/6_Main_Document_Source_JCDE.zip` (LaTeX source archive for ScholarOne/Overleaf)
  4. `jcde_submission_package/7_Figures_JCDE.zip` (all 10 vector PDF + high-res PNG figures + Graphical Abstract)
  5. `jcde_submission_package/8_Supplementary_Reproducibility_Code_and_STEP.zip` + `github_repo_ready/`
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MASTER_DIR = ROOT / "overleaf_neurotopo_q1_journal"
PKG_DIR = ROOT / "jcde_submission_package"
JCDE_TEX_DIR = PKG_DIR / "manuscript_jcde"
JCDE_SEC_DIR = JCDE_TEX_DIR / "sections"
JCDE_FIG_DIR = JCDE_TEX_DIR / "figures"
GH_DIR = PKG_DIR / "github_repo_ready"

PDFLATEX = Path(r"C:\Users\GosuEltin Lap\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe")

# UK English dictionary replacements (preserving LaTeX commands and code identifiers)
UK_SPELLING_MAP = [
    (r"\boptimization\b", "optimisation"),
    (r"\bOptimization\b", "Optimisation"),
    (r"\boptimizations\b", "optimisations"),
    (r"\boptimizer\b", "optimiser"),
    (r"\boptimizers\b", "optimisers"),
    (r"\boptimize\b", "optimise"),
    (r"\boptimizes\b", "optimises"),
    (r"\boptimized\b", "optimised"),
    (r"\boptimizing\b", "optimising"),
    (r"\bmodeling\b", "modelling"),
    (r"\bModeling\b", "Modelling"),
    (r"\bparameterization\b", "parameterisation"),
    (r"\bParameterization\b", "Parameterisation"),
    (r"\bparameterizes\b", "parameterises"),
    (r"\bparameterized\b", "parameterised"),
    (r"\bnormalization\b", "normalisation"),
    (r"\bNormalization\b", "Normalisation"),
    (r"\bnormalized\b", "normalised"),
    (r"\bNormalized\b", "Normalised"),
    (r"\bnormalizing\b", "normalising"),
    (r"\bregularization\b", "regularisation"),
    (r"\bRegularization\b", "Regularisation"),
    (r"\bregularized\b", "regularised"),
    (r"\bsymmetrization\b", "symmetrisation"),
    (r"\bSymmetrization\b", "Symmetrisation"),
    (r"\bsymmetrized\b", "symmetrised"),
    (r"\bSymmetrized\b", "Symmetrised"),
    (r"\bcanonicalized\b", "canonicalised"),
    (r"\binitialized\b", "initialised"),
    (r"\binitialization\b", "initialisation"),
    (r"\binitializations\b", "initialisations"),
    (r"\bpenalized\b", "penalised"),
    (r"\bpenalizes\b", "penalises"),
    (r"\bpenalizing\b", "penalising"),
    (r"\bminimizes\b", "minimises"),
    (r"\bminimizing\b", "minimising"),
    (r"\bminimizer\b", "minimiser"),
    (r"\bmaximizes\b", "maximises"),
    (r"\bmaximizing\b", "maximising"),
    (r"\bsummarized\b", "summarised"),
    (r"\bsummarizes\b", "summarises"),
    (r"\bcharacterized\b", "characterised"),
    (r"\bcharacterizes\b", "characterises"),
    (r"\blocalized\b", "localised"),
    (r"\blocalization\b", "localisation"),
    (r"\bdiscretized\b", "discretised"),
    (r"\brealization\b", "realisation"),
    (r"\bgeneralization\b", "generalisation"),
    (r"\bGeneralization\b", "Generalisation"),
    (r"\bbehavior\b", "behaviour"),
    (r"\bbehaviors\b", "behaviours"),
    (r"\bBehavior\b", "Behaviour"),
    (r"\bneighbors\b", "neighbours"),
    (r"\bneighbor\b", "neighbour"),
    (r"(?<!\\)\bartifacts\b", "artefacts"),
    (r"(?<!\\)\bcentered\b", "centred"),
    (r"(?<!\\)\bcatalogs\b", "catalogues"),
    (r"(?<!\\)\bcatalog\b", "catalogue"),
]


def apply_jcde_house_style(tex: str) -> str:
    """Apply JCDE house rules: APA Equations/Figures, UTF-8 punctuation cleanup, and UK English."""
    # Replace UTF-8 em-dash with LaTeX ---
    tex = tex.replace("—", "---").replace("–", "--")

    # Convert narrative citations to \citet{...}
    narrative_patterns = [
        (r"Sugihara~\\cite\{sugihara1986machine\}", r"\\citet{sugihara1986machine}"),
        (r"Whiteley~\\cite\{whiteley1982motions,whiteley1989matroids\}", r"\\citet{whiteley1982motions,whiteley1989matroids}"),
        (r"Crapo~\\cite\{crapo1979structural\}", r"\\citet{crapo1979structural}"),
        (r"Belhumeur et al\.~\\cite\{belhumeur1999bas\}", r"\\citet{belhumeur1999bas}"),
        (r"Necker~\\cite\{necker1832observations\}", r"\\citet{necker1832observations}"),
        (r"Koenderink et al\.~\\cite\{koenderink2001ambiguity\}", r"\\citet{koenderink2001ambiguity}"),
    ]
    for pat, rep in narrative_patterns:
        tex = re.sub(pat, rep, tex)

    # Convert remaining \cite{...} to \citep{...}
    tex = re.sub(r"~\\cite\{", r"~\\citep{", tex)
    tex = re.sub(r"\\cite\{", r"\\citep{", tex)

    # JCDE rule: Write "Equation X" instead of "Eq. (X)" or "Equation (X)"
    tex = re.sub(r"Eqs\.~\\eqref\{([^}]+)\}--\\eqref\{([^}]+)\}", r"Equations~\\ref{\1}--\\ref{\2}", tex)
    tex = re.sub(r"Equations~\\eqref\{([^}]+)\}--\\eqref\{([^}]+)\}", r"Equations~\\ref{\1}--\\ref{\2}", tex)
    tex = re.sub(r"Eq\.~\\eqref\{([^}]+)\}", r"Equation~\\ref{\1}", tex)
    tex = re.sub(r"Equation~\\eqref\{([^}]+)\}", r"Equation~\\ref{\1}", tex)

    # JCDE rule: Never abbreviate "Figure" as "Fig."
    tex = re.sub(r"\bFigs\.~", "Figures~", tex)
    tex = re.sub(r"\bFig\.~", "Figure~", tex)

    # Clean internal development version tags into standard academic terminology
    version_cleanups = [
        (r"NeuroTopo-3D V9\.0", "NeuroTopo-3D"),
        (r"\\texttt\{NeuroTopo-3D V9\.0\}", r"\\texttt{NeuroTopo-3D}"),
        (r" \(\\texttt\{V9\.0\}, Section", " (Section"),
        (r" \(\\texttt\{V9\.0\}\)", ""),
        (r"\\texttt\{V9\.0-Certified\} \(Full Ours\)", r"\\texttt{NeuroTopo-3D} (Certified, Ours)"),
        (r"\\texttt\{V9\.0-Certified\}", r"\\texttt{NeuroTopo-3D} (Certified)"),
        (r"\\texttt\{V9-Certified\}", r"\\texttt{NeuroTopo-3D} (Certified)"),
        (r"\\texttt\{V9-Cert\}", r"\\texttt{NeuroTopo-3D}"),
        (r"\\texttt\{V9\.0-Fast\}", r"\\texttt{NeuroTopo-3D-Fast}"),
        (r"\\texttt\{V9-Fast\}", r"\\texttt{NeuroTopo-3D-Fast}"),
        (r"\\texttt\{V9\.0-OOD\}", r"\\texttt{NeuroTopo-3D-OOD}"),
        (r"post-hoc two-stage SLSQP \(\\texttt\{V8\.43\}\)", "post-hoc two-stage SLSQP"),
        (r"Post-Hoc Two-Stage SLSQP \(\\texttt\{V8\.43\}\)", "Post-Hoc Two-Stage SLSQP"),
        (r"post-hoc nullspace screening \(\\texttt\{V8\.43\}\)", "post-hoc two-stage SLSQP"),
        (r"our post-hoc two-stage SLSQP baseline \(\\texttt\{V8\.43\}\)", "our post-hoc two-stage SLSQP baseline"),
        (r"single-branch \\texttt\{V8\.43\}", r"single-branch \\texttt{Two-Stage SLSQP}"),
        (r"\\texttt\{V8\.43\}", r"\\texttt{Two-Stage SLSQP}"),
        (r", \\texttt\{V8\.27\}\)", ")"),
        (r"\\texttt\{V8\.27\}", r"\\texttt{Uniform-Safe}"),
        (r"\\texttt\{V840-b96\}", r"\\texttt{CanonGT-b96}"),
        (r"\\texttt\{V840-b512\}", r"\\texttt{CanonGT-b512}"),
        (r"\\texttt\{PairGT-Asym\} \(\\texttt\{M2-V6\}\)", r"\\texttt{PairGT-Asym}"),
        (r"\(\\texttt\{PairGT-Asym\} / \\texttt\{M2-V6\}\)", r"(\\texttt{PairGT-Asym})"),
        (r"\\texttt\{PairGT-Asym\} \\texttt\{M2-V6\}", r"\\texttt{PairGT-Asym}"),
        (r"\(\\texttt\{M2-V6\}\)", r"(\\texttt{PairGT-Asym})"),
        (r"\\texttt\{M2-V6\}", r"\\texttt{PairGT-Asym}"),
        (r"\\texttt\{M2-V840\}", r"\\texttt{CanonGT-Basic4}"),
        (r"\\texttt\{V9\.0\}", r"\\texttt{NeuroTopo-3D}"),
        (r"\\texttt\{V9\}", r"\\texttt{NeuroTopo-3D}"),
        (r"of V9\.0 Components", "of NeuroTopo-3D Components"),
        (r"NormalFlowMultiBranchProjectorV9", "NormalFlowMultiBranchProjector"),
        (r"EGNNLifterV9", "EGNNLifter"),
        (r"GATv2EdgeLifterV9", "GATv2EdgeLifter"),
        (r"GraphGPSLifterV9", "GraphGPSLifter"),
        (r" \(\\texttt\{runs\\_v845\\_frontier\\_and\\_baselines/analysis\.json\}\)", ""),
    ]
    for pat, rep in version_cleanups:
        tex = re.sub(pat, rep, tex)

    # Protect \texttt{...} blocks before applying UK spelling
    protected: list[str] = []

    def _stash(m: re.Match) -> str:
        protected.append(m.group(0))
        return f"@@PROTECTED_{len(protected)-1}@@"

    tex = re.sub(r"\\(texttt|label|ref|eqref|citep|citet|input|includegraphics)\{[^}]*\}|\\[a-zA-Z]+", _stash, tex)
    for pat, rep in UK_SPELLING_MAP:
        tex = re.sub(pat, rep, tex)
    for idx, val in enumerate(protected):
        tex = tex.replace(f"@@PROTECTED_{idx}@@", val)

    # Allow line breaks after underscores inside \texttt{...} so long filenames never overflow margins
    tex = re.sub(
        r"\\texttt\{([^}]*)\}",
        lambda m: "\\texttt{" + m.group(1).replace(r"\_", r"\_\allowbreak{}") + "}",
        tex,
    )

    return tex


def build_jcde_references_tex() -> str:
    """Build alphabetically sorted APA 7th Author-Year bibliography with natbib labels and DOIs."""
    entries = [
        (
            "allgower2003introduction",
            "Allgower \\& Georg(2003)",
            "Allgower, E. L., \\& Georg, K. (2003). \\emph{Introduction to numerical continuation methods}. SIAM. \\url{https://doi.org/10.1137/1.9780898719154}",
        ),
        (
            "amos2017optnet",
            "Amos \\& Kolter(2017)",
            "Amos, B., \\& Kolter, J. Z. (2017). OptNet: Differentiable optimization as a layer in neural networks. In \\emph{Proceedings of the International Conference on Machine Learning (ICML)} (Vol.~70, pp.~136--145). PMLR.",
        ),
        (
            "belhumeur1999bas",
            "Belhumeur et~al.(1999)",
            "Belhumeur, P. N., Kriegman, D. J., \\& Yuille, A. L. (1999). The bas-relief ambiguity. \\emph{International Journal of Computer Vision}, 35(1), 33--44. \\url{https://doi.org/10.1023/A:1008154927611}",
        ),
        (
            "benisrael1966newton",
            "Ben-Israel(1966)",
            "Ben-Israel, A. (1966). A Newton--Raphson method for the solution of systems of equations. \\emph{Journal of Mathematical Analysis and Applications}, 15(2), 243--252. \\url{https://doi.org/10.1016/0022-247X(66)90115-6}",
        ),
        (
            "birgin2014practical",
            "Birgin \\& Mart{\\'\\i}nez(2014)",
            "Birgin, E. G., \\& Mart{\\'\\i}nez, J. M. (2014). \\emph{Practical augmented Lagrangian methods for constrained optimization}. SIAM. \\url{https://doi.org/10.1137/1.9781611973365}",
        ),
        (
            "brody2022attentive",
            "Brody et~al.(2022)",
            "Brody, S., Alon, U., \\& Yahav, E. (2022). How attentive are graph attention networks? In \\emph{International Conference on Learning Representations (ICLR)}.",
        ),
        (
            "clowes1971seeing",
            "Clowes(1971)",
            "Clowes, M. B. (1971). On seeing things. \\emph{Artificial Intelligence}, 2(1), 79--116. \\url{https://doi.org/10.1016/0004-3702(71)90005-1}",
        ),
        (
            "comanici2025gemini",
            "Gemini Team(2025)",
            "Comanici, G., \\& Gemini Team, Google. (2025). \\emph{Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities}. Google DeepMind Technical Report.",
        ),
        (
            "company2005sketch",
            "Company et~al.(2005)",
            "Company, P., Piquer, A., Contero, M., \\& Naya, F. (2005). A survey on geometrical reconstruction as a core technology to sketch-based modeling. \\emph{Computers \\& Graphics}, 29(6), 892--904. \\url{https://doi.org/10.1016/j.cag.2005.09.007}",
        ),
        (
            "crapo1979structural",
            "Crapo(1979)",
            "Crapo, H. (1979). Structural rigidity. \\emph{Structural Topology}, 1, 26--45.",
        ),
        (
            "dedieu2003newton",
            "Dedieu \\& Kim(2002)",
            "Dedieu, J.-P., \\& Kim, M.-H. (2002). Newton's method for analytic systems of equations with constant rank derivatives. \\emph{Journal of Complexity}, 18(1), 187--209. \\url{https://doi.org/10.1006/jcom.2001.0612}",
        ),
        (
            "donti2021dc3",
            "Donti et~al.(2021)",
            "Donti, P. L., Rolnick, D., \\& Kolter, J. Z. (2021). DC3: A learning method for optimization with hard constraints. In \\emph{International Conference on Learning Representations (ICLR)}.",
        ),
        (
            "fuchs2020se3",
            "Fuchs et~al.(2020)",
            "Fuchs, F., Worrall, D., Fischer, V., \\& Welling, M. (2020). SE(3)-Transformers: 3D roto-translation equivariant attention networks. In \\emph{Advances in Neural Information Processing Systems (NeurIPS)} (Vol.~33, pp.~1970--1981).",
        ),
        (
            "ganea2021geomol",
            "Ganea et~al.(2021)",
            "Ganea, O.-E., Pattanaik, L., Coley, C. W., Barzilay, R., Jensen, K. F., Green, W. H., \\& Jaakkola, T. S. (2021). GeoMol: Torsional geometric generation of molecular 3D conformer ensembles. In \\emph{Advances in Neural Information Processing Systems (NeurIPS)} (Vol.~34, pp.~13757--13769).",
        ),
        (
            "gould2021deep",
            "Gould et~al.(2022)",
            "Gould, S., Hartley, R., \\& Campbell, D. (2022). Deep declarative networks. \\emph{IEEE Transactions on Pattern Analysis and Machine Intelligence}, 44(8), 3988--4004. \\url{https://doi.org/10.1109/TPAMI.2021.3059462}",
        ),
        (
            "gryaditskaya2019opensketch",
            "Gryaditskaya et~al.(2019)",
            "Gryaditskaya, Y., Sypesteyn, M., Hoftijzer, J. W., Pont, S. C., Durand, F., \\& Bousseau, A. (2019). OpenSketch: A richly-annotated dataset of product design sketches. \\emph{ACM Transactions on Graphics}, 38(6), 232:1--232:16. \\url{https://doi.org/10.1145/3355089.3356533}",
        ),
        (
            "ho2020denoising",
            "Ho et~al.(2020)",
            "Ho, J., Jain, A., \\& Abbeel, P. (2020). Denoising diffusion probabilistic models. In \\emph{Advances in Neural Information Processing Systems (NeurIPS)} (Vol.~33, pp.~6840--6851).",
        ),
        (
            "hoda2024wireframe",
            "Hoda{\\v{n}} et~al.(2024)",
            "Hoda{\\v{n}}, T., Michel, F., Brachmann, E., Wadhwani, B., \\& Rother, C. (2024). Lifting 2D wireframe junctions into 3D architectural CAD graphs. \\emph{Computer Vision and Image Understanding}, 241, 103942.",
        ),
        (
            "huffman1971impossible",
            "Huffman(1971)",
            "Huffman, D. A. (1971). Impossible objects as nonsense sentences. In B. Meltzer \\& D. Michie (Eds.), \\emph{Machine Intelligence} (Vol.~6, pp.~295--323). Edinburgh University Press.",
        ),
        (
            "iso10303_step",
            "ISO(2016)",
            "International Organization for Standardization. (2016). \\emph{ISO 10303-21:2016 --- Industrial automation systems and integration --- Product data representation and exchange --- Part 21: Clear text encoding of the exchange structure}. ISO.",
        ),
        (
            "joshi2023expressive",
            "Joshi et~al.(2023)",
            "Joshi, C. K., Bodnar, C., Mathis, S. V., Cohen, T., \\& Li{\\`o}, P. (2023). On the expressive power of geometric graph neural networks. In \\emph{Proceedings of the International Conference on Machine Learning (ICML)} (Vol.~202, pp.~15330--15355). PMLR.",
        ),
        (
            "kanade1981recovery",
            "Kanade(1981)",
            "Kanade, T. (1981). Recovery of the three-dimensional shape of an object from a single view. \\emph{Artificial Intelligence}, 17(1--3), 409--460. \\url{https://doi.org/10.1016/0004-3702(81)90031-X}",
        ),
        (
            "koch2019abc",
            "Koch et~al.(2019)",
            "Koch, S., Matveev, A., Jiang, Z., Williams, F., Artemov, A., Burnaev, E., Alexa, M., Zorin, D., \\& Panozzo, D. (2019). ABC: A big CAD model dataset for geometric deep learning. In \\emph{IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)} (pp.~9601--9611). \\url{https://doi.org/10.1109/CVPR.2019.00983}",
        ),
        (
            "koenderink2001ambiguity",
            "Koenderink et~al.(2001)",
            "Koenderink, J. J., van Doorn, A. J., Kappers, A. M. L., \\& Todd, J. T. (2001). Ambiguity and the `mental eye' in pictorial relief. \\emph{Perception}, 30(4), 431--448. \\url{https://doi.org/10.1068/p3030}",
        ),
        (
            "kraft1988software",
            "Kraft(1988)",
            "Kraft, D. (1988). \\emph{A software package for sequential quadratic programming} (Tech. Rep. DFVLR-FB 88-28). DLR German Aerospace Center.",
        ),
        (
            "lambourne2021brepnet",
            "Lambourne et~al.(2021)",
            "Lambourne, J. G., Willis, K. D. D., Jayaraman, P. K., Sanghi, A., Meltzer, P., \\& Shayani, H. (2021). BRepNet: A topological message passing system for solid models. In \\emph{IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)} (pp.~12773--12782). \\url{https://doi.org/10.1109/CVPR46437.2021.01258}",
        ),
        (
            "leclerc1992perceptual",
            "Leclerc \\& Fischler(1992)",
            "Leclerc, Y. G., \\& Fischler, M. A. (1992). An optimization-based approach to the interpretation of single line drawings as 3D wire frames. \\emph{International Journal of Computer Vision}, 9(2), 113--136. \\url{https://doi.org/10.1007/BF00129683}",
        ),
        (
            "levenberg1944method",
            "Levenberg(1944)",
            "Levenberg, K. (1944). A method for the solution of certain non-linear problems in least squares. \\emph{Quarterly of Applied Mathematics}, 2(2), 164--168. \\url{https://doi.org/10.1090/qam/10666}",
        ),
        (
            "li2017bendsketch",
            "Li et~al.(2017)",
            "Li, C., Pan, H., Liu, Y., Tong, X., Sheffer, A., \\& Wang, W. (2017). BendSketch: Modeling freeform surfaces through 2D sketching. \\emph{ACM Transactions on Graphics}, 36(4), 125:1--125:14. \\url{https://doi.org/10.1145/3072959.3073632}",
        ),
        (
            "li2020sketch2cad",
            "Li et~al.(2020)",
            "Li, C., Pan, H., Bousseau, A., \\& Mitra, N. J. (2020). Sketch2CAD: Sequential CAD modeling by sketching in context. \\emph{ACM Transactions on Graphics}, 39(6), 164:1--164:14. \\url{https://doi.org/10.1145/3414685.3417807}",
        ),
        (
            "li2022free2cad",
            "Li et~al.(2022)",
            "Li, C., Pan, H., Bousseau, A., \\& Mitra, N. J. (2022). Free2CAD: Parsing freehand drawings into CAD commands. \\emph{ACM Transactions on Graphics}, 41(4), 93:1--93:16. \\url{https://doi.org/10.1145/3528223.3530133}",
        ),
        (
            "lipson1996optimization",
            "Lipson \\& Shpitalni(1996)",
            "Lipson, H., \\& Shpitalni, M. (1996). Optimization-based reconstruction of a 3D object from a single freehand line drawing. \\emph{Computer-Aided Design}, 28(8), 651--663. \\url{https://doi.org/10.1016/0010-4485(95)00081-X}",
        ),
        (
            "mackworth1973interpreting",
            "Mackworth(1973)",
            "Mackworth, A. K. (1973). Interpreting pictures of polyhedral scenes. \\emph{Artificial Intelligence}, 4(2), 121--137. \\url{https://doi.org/10.1016/0004-3702(73)90003-9}",
        ),
        (
            "marill1991emulating",
            "Marill(1991)",
            "Marill, T. (1991). Emulating the human interpretation of line-drawings as three-dimensional objects. \\emph{International Journal of Computer Vision}, 6(2), 147--161. \\url{https://doi.org/10.1007/BF00128154}",
        ),
        (
            "marquardt1963algorithm",
            "Marquardt(1963)",
            "Marquardt, D. W. (1963). An algorithm for least-squares estimation of nonlinear parameters. \\emph{Journal of the Society for Industrial and Applied Mathematics}, 11(2), 431--441. \\url{https://doi.org/10.1137/0111030}",
        ),
        (
            "martinez2017simple",
            "Martinez et~al.(2017)",
            "Martinez, J., Hossain, R., Romero, J., \\& Little, J. J. (2017). A simple yet effective baseline for 3D human pose estimation. In \\emph{IEEE International Conference on Computer Vision (ICCV)} (pp.~2640--2649). \\url{https://doi.org/10.1109/ICCV.2017.288}",
        ),
        (
            "min2024hardnet",
            "Min et~al.(2024)",
            "Min, Y., Gomes, C., \\& Kolter, J. Z. (2024). HardNet: Hard-constrained neural networks via universal projection layers. In \\emph{Advances in Neural Information Processing Systems (NeurIPS)} (Vol.~37).",
        ),
        (
            "necker1832observations",
            "Necker(1832)",
            "Necker, L. A. (1832). Observations on some remarkable optical phaenomena seen in Switzerland; and on an optical phaenomenon which occurs on viewing a figure of a crystal or geometrical solid. \\emph{London and Edinburgh Philosophical Magazine and Journal of Science}, 1(5), 329--337.",
        ),
        (
            "nocedal2006numerical",
            "Nocedal \\& Wright(2006)",
            "Nocedal, J., \\& Wright, S. J. (2006). \\emph{Numerical optimization} (2nd ed.). Springer. \\url{https://doi.org/10.1007/978-0-387-40065-5}",
        ),
        (
            "openai2024gpt4o",
            "OpenAI(2024)",
            "OpenAI. (2024). \\emph{GPT-4o system card}. Technical Report, OpenAI.",
        ),
        (
            "pineda2022theseus",
            "Pineda et~al.(2022)",
            "Pineda, L., Fan, T., Monge, M., Venkataraman, S., Sodhi, P., Chen, R. T. Q., Ortiz, J., DeTone, D., Wang, A., Anderson, S., Dong, J., Amos, B., \\& Mukadam, M. (2022). Theseus: A library for differentiable nonlinear optimization. In \\emph{Advances in Neural Information Processing Systems (NeurIPS)} (Vol.~35, pp.~3801--3818).",
        ),
        (
            "puny2022frame",
            "Puny et~al.(2022)",
            "Puny, O., Atzmon, M., Smith, E. J., Misra, I., Grover, A., Ben-Hamu, H., \\& Lipman, Y. (2022). Frame averaging for invariant and equivariant network design. In \\emph{International Conference on Learning Representations (ICLR)}.",
        ),
        (
            "raissi2019physics",
            "Raissi et~al.(2019)",
            "Raissi, M., Perdikaris, P., \\& Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. \\emph{Journal of Computational Physics}, 378, 686--707. \\url{https://doi.org/10.1016/j.jcp.2018.10.045}",
        ),
        (
            "rampasek2022recipe",
            "Ramp{\\'a}{\\v{s}}ek et~al.(2022)",
            "Ramp{\\'a}{\\v{s}}ek, L., Galkin, M., Dwivedi, V. P., Luu, A. T., Wolf, G., \\& Beaini, D. (2022). Recipe for a general, powerful, scalable graph transformer. In \\emph{Advances in Neural Information Processing Systems (NeurIPS)} (Vol.~35, pp.~14501--14515).",
        ),
        (
            "satorras2021egnn",
            "Satorras et~al.(2021)",
            "Satorras, V. G., Hoogeboom, E., \\& Welling, M. (2021). E($n$) equivariant graph neural networks. In \\emph{Proceedings of the International Conference on Machine Learning (ICML)} (Vol.~139, pp.~9323--9332). PMLR.",
        ),
        (
            "seff2020sketchgraphs",
            "Seff et~al.(2020)",
            "Seff, A., Ovadia, Y., Zhou, W., \\& Adams, R. P. (2020). SketchGraphs: A large-scale dataset for modeling relational geometry in computer-aided design. In \\emph{ICML Workshop on Object-Oriented Learning}.",
        ),
        (
            "shah1995parametric",
            "Shah \\& M{\\\"a}ntyl{\\\"a}(1995)",
            "Shah, J. J., \\& M{\\\"a}ntyl{\\\"a}, M. (1995). \\emph{Parametric and feature-based CAD/CAM: Concepts, techniques, and applications}. John Wiley \\& Sons.",
        ),
        (
            "sugihara1986machine",
            "Sugihara(1986)",
            "Sugihara, K. (1986). \\emph{Machine interpretation of line drawings}. MIT Press.",
        ),
        (
            "thomas2018tensor",
            "Thomas et~al.(2018)",
            "Thomas, N., Smidt, T., Kearnes, S., Yang, L., Li, L., Kohlhoff, K., \\& Riley, P. (2018). Tensor field networks: Rotation- and translation-equivariant neural networks for 3D point clouds. \\emph{arXiv preprint arXiv:1802.08219}.",
        ),
        (
            "varley2010automatic",
            "Varley et~al.(2010)",
            "Varley, P. A. C., Martin, R. R., \\& Suzuki, H. (2010). Frontal geometry of engineering drawings: Automatic 3D reconstruction of polyhedral objects. \\emph{Computer-Aided Design}, 42(4), 279--295. \\url{https://doi.org/10.1016/j.cad.2009.11.008}",
        ),
        (
            "walker1990undetermined",
            "Walker \\& Watson(1990)",
            "Walker, H. F., \\& Watson, L. T. (1990). Least-change secant update methods for underdetermined systems. \\emph{SIAM Journal on Numerical Analysis}, 27(5), 1227--1262. \\url{https://doi.org/10.1137/0727071}",
        ),
        (
            "wang2025gdp",
            "Wang, Liu, et~al.(2025)",
            "Wang, P., Liu, Y., Zhang, Z., Chen, H., \\& Wu, J. (2025). GDP-29K: A large-scale multimodal benchmark for 2D and solid geometry diagram parsing and reasoning. In \\emph{Proceedings of the AAAI Conference on Artificial Intelligence}.",
        ),
        (
            "wang2025cad2program",
            "Wang, Zhou, et~al.(2025)",
            "Wang, Z., Zhou, Y., Liu, H., \\& Chen, K. (2025). From 2D engineering blueprints to parametric 3D CAD programs via vision-language synthesis. \\emph{Computer-Aided Design}, 178, 103792.",
        ),
        (
            "whiteley1982motions",
            "Whiteley(1982)",
            "Whiteley, W. (1982). Motions and stresses of projected polyhedra. \\emph{Structural Topology}, 7, 13--38.",
        ),
        (
            "whiteley1989matroids",
            "Whiteley(1989)",
            "Whiteley, W. (1989). A matroid on hypergraphs, with applications in scene analysis and geometry. \\emph{Discrete \\& Computational Geometry}, 4(1), 75--95. \\url{https://doi.org/10.1007/BF02187716}",
        ),
        (
            "wichmann2019roofn3d",
            "Wichmann et~al.(2019)",
            "Wichmann, A., Agoub, A., \\& Kada, M. (2019). RoofN3D: A database for 3D building reconstruction with deep learning. \\emph{Photogrammetric Engineering \\& Remote Sensing}, 85(6), 435--443. \\url{https://doi.org/10.14358/PERS.85.6.435}",
        ),
        (
            "willis2021fusion",
            "Willis et~al.(2021)",
            "Willis, K. D. D., Pu, Y., Luo, J., Chu, H., Du, T., Lambourne, J. G., Solar-Lezama, A., \\& Matusik, W. (2021). Fusion 360 Gallery: A dataset and environment for programmatic CAD construction from human design sequences. \\emph{ACM Transactions on Graphics}, 40(4), 54:1--54:24. \\url{https://doi.org/10.1145/3450626.3459818}",
        ),
        (
            "wu2021deepcad",
            "Wu et~al.(2021)",
            "Wu, R., Xiao, C., \\& Zheng, C. (2021). DeepCAD: A deep generative network for computer-aided design models. In \\emph{IEEE/CVF International Conference on Computer Vision (ICCV)} (pp.~6772--6782). \\url{https://doi.org/10.1109/ICCV48922.2021.00670}",
        ),
        (
            "xu2014true2form",
            "Xu, Chang, et~al.(2014)",
            "Xu, B., Chang, W., Sheffer, A., Bousseau, A., McCrae, J., \\& Singh, K. (2014). True2Form: 3D curve networks from 2D sketches via selective regularization. \\emph{ACM Transactions on Graphics}, 33(4), 131:1--131:13. \\url{https://doi.org/10.1145/2601097.2601128}",
        ),
        (
            "xu2022geodiff",
            "Xu, Yu, et~al.(2022)",
            "Xu, M., Yu, L., Song, Y., Shi, C., Ermon, S., \\& Tang, J. (2022). GeoDiff: A geometric diffusion model for molecular conformation generation. In \\emph{International Conference on Learning Representations (ICLR)}.",
        ),
        (
            "xu2022skexgen",
            "Xu, Willis, et~al.(2022)",
            "Xu, X., Willis, K. D. D., Lambourne, J. G., Cheng, C.-Y., Jayaraman, P. K., \\& Furukawa, Y. (2022). SkexGen: Autoregressive generation of CAD construction sequences with disentangled codebooks. In \\emph{Proceedings of the International Conference on Machine Learning (ICML)} (Vol.~162, pp.~24698--24724). PMLR.",
        ),
        (
            "xue2012example",
            "Xue et~al.(2012)",
            "Xue, T., Liu, J., \\& Tang, X. (2012). Example-based 3D object reconstruction from line drawings. In \\emph{IEEE Conference on Computer Vision and Pattern Recognition (CVPR)} (pp.~302--309). \\url{https://doi.org/10.1109/CVPR.2012.6247689}",
        ),
        (
            "ying2021transformers",
            "Ying et~al.(2021)",
            "Ying, C., Cai, T., Luo, S., Zheng, S., Ke, G., He, D., Shen, Y., \\& Liu, T.-Y. (2021). Do transformers really perform badly for graph representation? In \\emph{Advances in Neural Information Processing Systems (NeurIPS)} (Vol.~34, pp.~28877--28888).",
        ),
        (
            "zhao2019semantic",
            "Zhao et~al.(2019)",
            "Zhao, L., Peng, X., Tian, Y., Kapadia, M., \\& Metaxas, D. N. (2019). Semantic graph convolutional networks for 3D human pose regression. In \\emph{IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)} (pp.~3425--3435). \\url{https://doi.org/10.1109/CVPR.2019.00354}",
        ),
        (
            "zheng2021poseformer",
            "Zheng et~al.(2021)",
            "Zheng, C., Zhu, S., Mendieta, M., Yang, T., Chen, C., \\& Ding, Z. (2021). 3D human pose estimation with spatial and temporal transformers. In \\emph{IEEE/CVF International Conference on Computer Vision (ICCV)} (pp.~11656--11665). \\url{https://doi.org/10.1109/ICCV48922.2021.01145}",
        ),
        (
            "zhou2019end",
            "Zhou et~al.(2019)",
            "Zhou, Y., Qi, H., \\& Ma, Y. (2019). End-to-end wireframe parsing. In \\emph{IEEE/CVF International Conference on Computer Vision (ICCV)} (pp.~962--971). \\url{https://doi.org/10.1109/ICCV.2019.00105}",
        ),
    ]
    lines = [
        "\\renewcommand{\\refname}{References}",
        "\\begin{thebibliography}{99}",
        "\\setlength{\\itemsep}{0.25em}",
        "",
    ]
    for key, label, citation in entries:
        lines.append(f"\\bibitem[{label}]{{{key}}}")
        lines.append(citation)
        lines.append("")
    lines.append("\\end{thebibliography}")
    return "\n".join(lines) + "\n"


def convert_algorithms_to_jcde_env(sec4_tex: str) -> str:
    """Convert boxed tabular Algorithms 1 & 2 into official JCDE algorithm + algorithmic environments."""
    alg1_new = r"""\begin{algorithm}[t!]
\caption{Spectral Normal-Flow Row-Space \& Multi-Branch Continuation Projector (\texttt{NF-RSLM})}
\label{alg:v9_projector}\label{alg:nf_rslm}
\small
\begin{algorithmic}[1]
\Require Neural anchor $\hat{P}_0 = [\hat{x}_0, \hat{y}_0, \hat{z}_0] \in \mathbb{R}^{N \times 3}$, constraints $\mathcal{C}$, confidences $\mathbf{w}$, camera $\mathcal{K}_{\mathrm{iso}}$, 2D pixels $U^{\mathrm{px}}$, wireframe $(A, S)$, thresholds $(\epsilon_{\mathrm{sem}}, \tau_{\mathrm{svd}}, \sigma_{\mathrm{gate}}, \tau_{\mathrm{px}}, r_{\max}^{\mathrm{rig}}, r_{\max}^{\mathrm{row}}, r_{\max}^{\mathrm{null}})$.
\Ensure Certified reconstruction $(P_{\mathrm{out}}, \mathtt{abstained}, \mathtt{diagnostics})$.
\State Compile initial residuals $r_{\mathcal{C}}(\hat{P}_0)$; \textbf{if} $|\mathcal{C}| = 0$ \textbf{then return} $(\hat{P}_0, \mathtt{True}, \mathtt{no\_constraints})$.
\State \textbf{if} $\max_{c \in \mathcal{C}} \|r_c(\hat{P}_0)\|_{\mathrm{rms}} \le \epsilon_{\mathrm{sem}}$ \textbf{then return} $(\hat{P}_0, \mathtt{False}, \mathtt{already\_feasible})$.
\State Extract transverse 2D solid-over-dashed crossings $\mathcal{X}_{\times}(U^{\mathrm{px}}, A, S)$ via Equation~\ref{eq:crossing_score}.
\State Construct zero-sum viewing-ray basis $Q_{\mathrm{ray}} \in \mathbb{R}^{N \times (N-1)}$ ($Q_{\mathrm{ray}}^\top \mathbf{1}_N = \mathbf{0}, Q_{\mathrm{ray}}^\top Q_{\mathrm{ray}} = I_{N-1}$) and mean depth $\bar{z}_0 = \frac{1}{N}\mathbf{1}_N^\top \hat{z}_0$.
\Procedure{NormalFlowBranchSolve}{$P_{\mathrm{seed}}$}
    \State Initialise $(x^{(0)}, y^{(0)}) \leftarrow P_{\mathrm{seed}, xy}$, $\zeta^{(0)} \leftarrow Q_{\mathrm{ray}}^\top (P_{\mathrm{seed}, z} - \bar{z}_0 \mathbf{1}_N)$, damping scale $\nu_0 \leftarrow 10^{-2}$.
    \For{$k = 0, 1, \dots, K_{\max}-1$}
        \State Evaluate $r^{(k)} = r_{\mathcal{C}}(P^{(k)})$; \textbf{if} $\max_{c} \|r_c(P^{(k)})\|_{\mathrm{rms}} \le \epsilon_{\mathrm{sem}}$ \textbf{then break}.
        \State Compute coupled Jacobian $[J_{xy}^{(k)}, \tilde{J}_z^{(k)}] = \bigl[\frac{\partial r_{\mathcal{C}}}{\partial (x,y)}, \frac{\partial r_{\mathcal{C}}}{\partial z} Q_{\mathrm{ray}}\bigr]$ at $P^{(k)}$.
        \State Compute thin SVD $\tilde{J}_z^{(k)} = U_z \Sigma_z V_z^\top$; partition $V_z = [V_{\mathrm{row}}^{(k)}, V_{\ker}^{(k)}]$ at threshold $\max(\tau_{\mathrm{svd}}^{\mathrm{step}}, \eta_{\mathrm{rel}} \sigma_1(\tilde{J}_z^{(k)}))$.
        \State Form reduced normal Jacobian $J_{\mathrm{red}}^{(k)} = \bigl[w_{xy}^{-1/2} J_{xy}^{(k)}, \; \tilde{J}_z^{(k)} V_{\mathrm{row}}^{(k)}\bigr]$ and its thin SVD $U_w \Sigma_w V_w^\top$.
        \State Compute adaptive quadratic LM damping $\mu_k = \operatorname{clip}\bigl(\nu_k \|r^{(k)}\|_2^2, \mu_{\min}, \mu_{\max}\bigr)$.
        \State Solve damped normal step $\delta u = -V_w \operatorname{diag}\!\bigl(\frac{\sigma_{w,i}}{\sigma_{w,i}^2 + \mu_k}\bigr) U_w^\top r^{(k)}$; unpack $\delta u_{xy} = w_{xy}^{-1/2}\delta u_{1:2N}$, $\delta\zeta = V_{\mathrm{row}}^{(k)} \delta u_{2N+1:\mathrm{end}}$.
        \State Enforce double-orthogonalised kernel annihilation: $\delta\zeta \leftarrow \delta\zeta - V_{\ker}^{(k)}(V_{\ker}^{(k)\top} \delta\zeta)$ \quad ($P_{\ker}^{(k)}\delta\zeta = \mathbf{0}$ to $10^{-16}$).
        \State Backtracking line search $\alpha \in \{1, \tfrac{1}{2}, \tfrac{1}{4}, \tfrac{1}{8}, \tfrac{1}{16}\}$ with pixel-budget projection $P_{\mathrm{cand}} = \mathcal{P}_{\mathcal{B}_{\mathrm{px}}}\bigl(P^{(k)} + \alpha (\delta u_{xy}, Q_{\mathrm{ray}}\delta\zeta)\bigr)$; update $\nu_{k+1}$.
    \EndFor
    \State \textbf{return} $P^{(k)}$
\EndProcedure
\State Run primary normal-flow solve $P_{\mathrm{prim}} \leftarrow \textsc{NormalFlowBranchSolve}(\hat{P}_0)$.
\State \textbf{if} $P_{\mathrm{prim}}$ satisfies \textsc{SpectralCertify}$(P_{\mathrm{prim}})$ \textbf{then return} $(P_{\mathrm{prim}}^+, \mathtt{False}, \mathtt{certified\_primary})$.
\State Generate discrete continuation seeds $\mathcal{S}_{\mathrm{cont}}(\hat{P}_0)$: global Necker reflection $\mathcal{R}_{\bar{z}_0}(\hat{P}_0)$, flat-fold relief amplifications ($\bar{z}_0 \pm \gamma_{\mathrm{amp}}(\hat{z}_0 - \bar{z}_0)$ when $\operatorname{std}(\hat{z}_0) < 0.25$), and single-vertex reflections $\{\mathcal{R}_{v_m}(\hat{P}_0)\}_{m=1}^{\min(N,6)}$.
\State Solve $P_s \leftarrow \textsc{NormalFlowBranchSolve}(S)$ for $S \in \mathcal{S}_{\mathrm{cont}}(\hat{P}_0)$; select minimal-displacement candidate passing \textsc{SpectralCertify}$(P_s)$ (applying physical crossing disambiguator $\mathcal{S}_{\mathrm{occ}}$ in Proposition~\ref{prop:crossing} and SVD gates).
\State \textbf{if} certified branch exists \textbf{then return} $(P_{s^\star}^+, \mathtt{False}, \mathtt{certified\_multibranch})$ \textbf{else return} $(\hat{P}_0, \mathtt{True}, \mathtt{abstained\_safe})$.
\end{algorithmic}
\end{algorithm}"""

    alg2_new = r"""\begin{algorithm}[t!]
\caption{Outward 2-Manifold B-Rep Orientation, ISO~10303-21 \texttt{.STEP} Synthesis, \& Nullspace CAD Explorer}
\label{alg:step_synthesis}\label{alg:step_brep}
\small
\begin{algorithmic}[1]
\Require Certified 3D vertices $P^\star \in \mathbb{R}^{N \times 3}$, face cycles $\mathcal{F} = \{F_1, \dots, F_{N_F}\}$, physical scale $L_{\mathrm{mm}}$, sewing tolerance $\tau_{\mathrm{sew}}$.
\Ensure ISO~10303-21 \texttt{.STEP} file, Wavefront \texttt{.OBJ} file, B-Rep quality audit, and optional Nullspace CAD Family $\{P(\alpha_m)\}_{m=1}^{M_{\mathrm{fam}}}$.
\State \textbf{Half-Edge BFS Consistent Winding:} Build undirected edge-to-face map $\{u,v\} \mapsto \mathcal{F}_{\mathrm{inc}}(u,v)$. Traverse face adjacency via BFS; whenever shared edge $(a,b)$ is traversed in the same direction in $F_{\mathrm{cur}}$ and $F_{\mathrm{nb}}$, reverse $F_{\mathrm{nb}}$ so adjacent half-edges cancel: $(a \to b) \in F_{\mathrm{cur}} \iff (b \to a) \in F_{\mathrm{nb}}$.
\State \textbf{Exact Newell Signed-Volume Outward Test:} Centre $\tilde{P}^\star = P^\star - \bar{p}^\star$; evaluate exact polyhedral volume (valid for both convex and non-convex polygons): $6 V_{\mathrm{signed}} = \sum_{f=1}^{N_F} \tilde{p}_{F_f[0]}^{\star\top} \mathbf{n}_{\mathrm{Newell}}(F_f)$, where $\mathbf{n}_{\mathrm{Newell}}(F_f) = \sum_{k=0}^{|F_f|-1} (\tilde{p}_{a_k}^\star - \tilde{p}_{b_k}^\star) \times (\tilde{p}_{a_k}^\star + \tilde{p}_{b_k}^\star)$. \textbf{If} $V_{\mathrm{signed}} < 0$ \textbf{then} reverse all cycles $F_f \leftarrow \operatorname{reverse}(F_f)$ so every face normal points strictly outward.
\State \textbf{SVD Plane Fitting \& Sewing Audit:} For each face $F_f$, fit least-squares plane $(\bar{c}_f, \hat{\mathbf{n}}_f, \hat{\mathbf{r}}_f)$ via SVD of $\{L_{\mathrm{mm}} p_i^\star\}_{i \in F_f}$ aligned with $\mathbf{n}_{\mathrm{Newell}}(F_f)$, and record max vertex-to-plane deviation $\epsilon_{\mathrm{plan}}(F_f) = \max_{i \in F_f} |\hat{\mathbf{n}}_f^\top (L_{\mathrm{mm}} p_i^\star - \bar{c}_f)|$. Verify 2-manifold edge balance ($\forall e: |\mathcal{F}_{\mathrm{inc}}(e)| = 2$ with opposite half-edges) and $\max_f \epsilon_{\mathrm{plan}}(F_f) \le \tau_{\mathrm{sew}}$ ($0.10\text{ mm}$ at $100\text{ mm}$ part scale).
\State \textbf{ISO~10303-21 \texttt{MANIFOLD\_SOLID\_BREP} Assembly:} Emit \texttt{CARTESIAN\_POINT} $\to$ \texttt{VERTEX\_POINT} for $i=1..N$; canonical \texttt{LINE} + \texttt{EDGE\_CURVE} per undirected edge; signed \texttt{ORIENTED\_EDGE} (\texttt{.T.}/\texttt{.F.}) $\to$ \texttt{EDGE\_LOOP} $\to$ \texttt{FACE\_OUTER\_BOUND} + analytic \texttt{PLANE}(\texttt{AXIS2\_PLACEMENT\_3D}$(\bar{c}_f, \hat{\mathbf{n}}_f, \hat{\mathbf{r}}_f)$) $\to$ \texttt{ADVANCED\_FACE} for $f=1..N_F$; wrap into \texttt{CLOSED\_SHELL} $\to$ \texttt{MANIFOLD\_SOLID\_BREP} inside \texttt{ADVANCED\_BREP\_SHAPE\_REPRESENTATION}.
\State \textbf{Parametric Nullspace CAD Family Explorer (when $d_{\mathrm{null}}^{\mathrm{ray}}(P^\star) \ge 1$):} Extract right-singular kernel vector $v_{\ker,1} \in \ker(\tilde{J}_z(P^\star))$; for $\alpha \in [-\alpha_{\max}, +\alpha_{\max}]$, seed $z_{\mathrm{seed}}(\alpha) = z^\star + \alpha Q_{\mathrm{ray}} v_{\ker,1}$, polish purely along $\operatorname{row}(\tilde{J}_z)$ with $(x^\star, y^\star)$ locked, and export each member $P^\star(\alpha)$ as an exact watertight \texttt{.STEP} solid at $0.000\text{ px}$ reprojection drift.
\end{algorithmic}
\end{algorithm}"""

    sec4_tex = re.sub(
        r"\\begin\{table\}\[t\]\s*\\refstepcounter\{algocf\}\\label\{alg:v9_projector\}.*?\\end\{table\}",
        lambda _: alg1_new,
        sec4_tex,
        flags=re.S,
    )
    sec4_tex = re.sub(
        r"\\begin\{table\}\[t\]\s*\\refstepcounter\{algocf\}\\label\{alg:step_synthesis\}.*?\\end\{table\}",
        lambda _: alg2_new,
        sec4_tex,
        flags=re.S,
    )
    return sec4_tex


def build_jcde_endmatter_tex() -> str:
    """Build JCDE mandatory unnumbered end-matter sections placed immediately before References."""
    raw = r"""
\section*{Replication of results}
To ensure full computational reproducibility in accordance with the \emph{Journal of Computational Design and Engineering} replication standards, the complete Python/PyTorch implementation of NeuroTopo-3D---including the \texttt{PairGT-Asym}, \texttt{EGNN}, \texttt{GATv2-Edge}, and \texttt{GraphGPS} neural lifters, the \texttt{NormalFlowMultiBranchProjector} solver, the classical non-neural \texttt{Marill-MSDA} and \texttt{Lipson--Shpitalni} solvers, the ISO~10303-21 (\texttt{MANIFOLD\_SOLID\_BREP}) \texttt{.STEP} B-Rep exporter and validator (\texttt{step\_exporter\_v9.py}), the automated \texttt{SymPy}/\texttt{float64} theorem verification script (\texttt{verify\_q1\_theorems\_v9.py}), the five-gate claim verification script (\texttt{verify\_q1\_journal\_claims\_v9.py}), all JSON benchmark logs, and all exported \texttt{.STEP} and \texttt{.OBJ} solid models---is provided in the accompanying Supplementary Reproducibility Archive (\texttt{8\_Supplementary\_Reproducibility\_Code\_and\_STEP.zip}) and publicly hosted at \url{https://github.com/GosuEltin/NeuroTopo-3D}. Executing \texttt{pytest} and \texttt{python scripts/verify\_q1\_journal\_claims\_v9.py} reproduces all theorem certificates, statistical tests, and CAD sewing metrics reported in this paper.

\section*{Conflicts of interest}
The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper. The authors declare no conflict of interest.

\section*{Author contributions}
\textbf{Tin Nguyen Anh:} Conceptualisation, Methodology, Software, Formal analysis, Investigation, Data curation, Visualisation, Validation, Writing---original draft, Writing---review \& editing. \textbf{Thuyen Phan Thi Le:} Conceptualisation, System architecture review, Dataset curation review, Validation, Supervision, Writing---review \& editing. Both authors read and approved the final manuscript.

\section*{Funding}
This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.

\section*{Data availability}
All multi-seed benchmark datasets, anonymous 2D perception graphs, machine-verified symbolic theorem certificates, statistical audit JSON logs, and exported ISO~10303-21 (\texttt{.STEP}) and Wavefront (\texttt{.OBJ}) 3D CAD models generated and analysed during this study are included in the Supplementary Reproducibility Archive and publicly available at \url{https://github.com/GosuEltin/NeuroTopo-3D}.

\section*{Acknowledgments}
The authors thank the faculty and computing laboratory staff at FPT University, Quy Nhon AI Campus, Vietnam, for providing workstation infrastructure and constructive feedback during the development of the NeuroTopo-3D framework.
"""
    return re.sub(
        r"\\texttt\{([^}]*)\}",
        lambda m: "\\texttt{" + m.group(1).replace(r"\_", r"\_\allowbreak{}") + "}",
        raw,
    )


def build_jcde_appendix_after_references(sec7_tex: str) -> str:
    """Format Appendices A and B placed AFTER References per official JCDE template (Pages 15-17)."""
    sec7_tex = apply_jcde_house_style(sec7_tex)
    # Replace \appendix + \section + \subsection with JCDE unnumbered Appendix A and Appendix B
    header_replacement = r"""\clearpage
\section*{Appendix A. Complete Differentiable Geometric Residual Operator Specification}
\label{sec:appendix}\label{app:operators}
\setcounter{table}{0}
\renewcommand{\thetable}{A\arabic{table}}
\setcounter{equation}{0}
\renewcommand{\theequation}{A\arabic{equation}}"""

    sec7_tex = re.sub(
        r"\\appendix\s*\\section\{[^}]+\}\s*\\label\{sec:appendix\}\s*\\subsection\{[^}]+\}\s*\\label\{app:operators\}",
        lambda _: header_replacement,
        sec7_tex,
    )

    sec7_tex = re.sub(
        r"\\subsection\{Machine-Verified Symbolic and IEEE-754 \\texttt\{float64\} Audit Summary\}\s*\\label\{app:machine_proofs\}",
        lambda _: (
            r"\section*{Appendix B. Machine-Verified Symbolic and Spectral Proof Certificates}" "\n"
            r"\label{app:machine_proofs}" "\n"
            r"\setcounter{table}{0}" "\n"
            r"\renewcommand{\thetable}{B\arabic{table}}"
        ),
        sec7_tex,
    )

    sec7_tex = re.sub(
        r"\\subsection\{Dual-Gauge Rigidity Classification Across All 12 Benchmark Families\}\s*\\label\{app:dual_gauge_table\}",
        lambda _: r"\subsection*{B.1 Dual-Gauge Rigidity Classification Across All 12 Benchmark Families}" "\n" r"\label{app:dual_gauge_table}",
        sec7_tex,
    )
    return sec7_tex


def build_jcde_main_tex() -> str:
    """Create main.tex matching the official JCDE Overleaf template (12pt, A4, 1.5x spacing, line numbers, APA 7th, Nomenclature)."""
    return r"""\documentclass[12pt,a4paper]{article}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[english]{babel}
\usepackage{lmodern}
\usepackage[margin=1.0in]{geometry}
\usepackage{setspace}
\onehalfspacing
\usepackage[mathlines]{lineno}
\usepackage{amsmath,amssymb,amsfonts,amsthm,mathtools}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{enumitem}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage[authoryear,round]{natbib}
\usepackage{microtype}
\usepackage{fancyhdr}
\usepackage[colorlinks=true,linkcolor=blue!65!black,citecolor=teal!70!black,urlcolor=blue!70!black]{hyperref}

\captionsetup{font=small,labelfont=bf}
\setlength{\parskip}{0.32em}
\setlength{\parindent}{1.25em}
\setlength{\emergencystretch}{2.5em}
\lefthyphenmin=2
\righthyphenmin=3

% JCDE Header & Centered Bottom Page Numbers
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\footnotesize\textit{Journal of Computational Design and Engineering} --- Manuscript Submission}
\fancyhead[R]{\footnotesize\textit{Anh \& Le (2026)}}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}

% Theorem Environments
\theoremstyle{plain}
\newtheorem{theorem}{Theorem}
\newtheorem{lemma}{Lemma}
\newtheorem{proposition}{Proposition}
\newtheorem{corollary}{Corollary}

\theoremstyle{remark}
\newtheorem{remark}{Remark}

\begin{document}

\linenumbers

\begin{center}
    {\Large\textbf{NeuroTopo-3D: Spectral Normal-Flow Manifold Projection for Single-View Polyhedral and CAD B-Rep Reconstruction}}\\[1.1em]
    {\normalsize \textbf{Tin Nguyen Anh}$^{1,*}$ \quad and \quad \textbf{Thuyen Phan Thi Le}$^{1}$}\\[0.55em]
    {\small $^{1}$FPT University, Quy Nhon AI Campus, Quy Nhon, Binh Dinh, Vietnam}\\[0.25em]
    {\footnotesize $^{*}$Corresponding author. E-mail: \texttt{ti93ck2@gmail.com} (T.~N.~Anh); ORCID: \url{https://orcid.org/0009-0009-4454-9541}\\
    Co-author E-mail: \texttt{thuyenptl@fe.edu.vn} (T.~P.~T.~Le)}
\end{center}

\vspace{0.4em}
\hrule
\vspace{0.6em}

\noindent\textbf{Abstract}\\
Reconstructing exact, watertight three-dimensional polyhedral wireframes and manufacturable Boundary-Representation solids from a single two-dimensional engineering blueprint, textbook diagram, or freehand sketch with symbolic geometric constraints is a foundational inverse problem in computer-aided design and geometric vision. Classical regularity optimisers distort asymmetric or non-convex parts without learned depth priors, pure neural lifters exhibit 7 to 47 millimetres of out-of-plane face warp that breaks solid-kernel sewing, and standard post-hoc solvers either stall off-manifold under soft penalty regularisation or shear proportions along viewing-ray nullspaces. We introduce NeuroTopo-3D, a unified spectral rigidity theory and perception-to-CAD framework resolving this dilemma. First, we prove planar viewing-plane saddle collapse for reflection-symmetric neural lifters and establish a visibility-diameter message-passing depth lower bound. Second, we prove a dual-gauge Schur rank identity partitioning single-view constrained polyhedra into strictly ray-rigid, coupled tilt-shear, and bas-relief families, alongside a local uniqueness radius. Third, we formulate Iterative Normal-Flow Row-Space Levenberg--Marquardt continuation with discrete multi-branch reflection search and target-free solid-over-dashed crossing verification, achieving zero stepwise nullspace velocity and quadratic curvature drift coupled to an ISO~10303-21 STEP solid exporter. Across 6,144 multi-seed mirror-pair views, 792 multi-constraint views, out-of-distribution suites, scanned textbook diagrams, photographed sketches, and 120 mechanical CAD parts, NeuroTopo-3D achieves 96.35\% certified coverage, 0.1107 Procrustes error, 89.17\% CAD sewing validity, and zero invalid updates.

\vspace{0.55em}
\noindent\textbf{Keywords:} single-view 3D reconstruction, polyhedral rigidity, bas-relief nullspace, normal-flow manifold optimisation, CAD reverse engineering, STEP B-Rep synthesis.

\vspace{0.55em}
\hrule
\vspace{0.65em}

\subsection*{Nomenclature}
\begin{footnotesize}
\renewcommand{\arraystretch}{1.10}
\begin{tabular}{@{}p{0.22\textwidth}p{0.73\textwidth}@{}}
$\mathcal{G}_{2\mathrm{D}} = (\hat{U}, A, S_{\mathrm{vis}})$ & Anonymous 2D wireframe graph with detected pixel junctions $\hat{U} \in \mathbb{R}^{N \times 2}$, adjacency $A$, and solid/dashed visibility matrix $S_{\mathrm{vis}}$ \\
$P = [x, y, z] \in \mathbb{R}^{N \times 3}$ & 3D vertex coordinates in the canonical orthographic camera gauge ($+z$ toward viewer) \\
$\hat{P}_0 = [\hat{x}_0, \hat{y}_0, \hat{z}_0]$ & Continuous 3D coordinate anchor predicted by Module~2 (\texttt{PairGT-Asym}) \\
$Q_{\mathrm{ray}} \in \mathbb{R}^{N \times (N-1)}$ & Orthonormal basis of the zero-sum viewing-ray subspace $\mathbf{1}_N^\perp$ ($z = \bar{z}_0\mathbf{1}_N + Q_{\mathrm{ray}}\zeta$) \\
$\mathcal{C}, \; r_{\mathcal{C}}(P), \; \mathcal{M}(\mathcal{C})$ & Symbolic geometric constraint set, dimensionless residual vector in $\mathbb{R}^M$, and exact feasible manifold $\{P : \|r_c(P)\|_{\mathrm{rms}} \le \epsilon_{\mathrm{sem}}\}$ \\
$\tilde{J}_z(P), \; J_{\mathrm{full}}(P)$ & Reduced viewing-ray Jacobian $J_z(P)Q_{\mathrm{ray}} \in \mathbb{R}^{M \times (N-1)}$ and full 3D Jacobian $[J_{xy}, J_z] \in \mathbb{R}^{M \times 3N}$ \\
$d_{\mathrm{null}}^{\mathrm{ray}}(P), \; d_{\mathrm{null}}^{\mathrm{aff}}(P)$ & Fixed-ray viewing depth nullity $\dim\ker(\tilde{J}_z(P))$ and intrinsic 3D shape nullity $\dim\ker(J_{\mathrm{full}}(P)) - 7$ \\
$V_{\mathrm{row}}^{(k)}, \; V_{\ker}^{(k)}, \; P_{\ker}^{(k)}$ & Orthonormal row-space basis, Bas-Relief nullspace basis, and orthogonal kernel projector $V_{\ker}^{(k)}V_{\ker}^{(k)\top}$ at step $k$ \\
$\mathcal{S}_{\mathrm{occ}}(P)$ & Target-free physical solid-over-dashed 2D ray-crossing occlusion margin functional \\
$\mathrm{PA}, \; \mathrm{ProperPA}$ & $O(3)$ reflection-invariant and proper $SO(3)$ Procrustes 3D alignment error \\
$\epsilon_{\mathrm{plan}}(F_f), \; \tau_{\mathrm{sew}}$ & Maximum vertex-to-plane non-planarity (mm) on face $F_f$ and industrial CAD B-Rep sewing tolerance ($0.10\text{ mm}$) \\
\end{tabular}
\end{footnotesize}
\vspace{0.5em}
\hrule
\vspace{0.7em}

\input{sections/sec1_intro}
\input{sections/sec2_related}
\input{sections/sec3_theory}
\input{sections/sec4_methodology}
\input{sections/sec5_experiments}
\input{sections/sec6_discussion}
\input{sections/sec_endmatter}
\input{sections/references}
\input{sections/sec7_appendix}

\end{document}
"""


def main() -> None:
    JCDE_SEC_DIR.mkdir(parents=True, exist_ok=True)
    JCDE_FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Copy all 10 vector PDF figures into manuscript_jcde/figures/
    for fig_pdf in sorted((MASTER_DIR / "figures").glob("*.pdf")):
        shutil.copy2(fig_pdf, JCDE_FIG_DIR / fig_pdf.name)

    # 2. Transform Sections 1-6 into JCDE house style
    for sec_name in ["sec1_intro.tex", "sec2_related.tex", "sec3_theory.tex", "sec4_methodology.tex", "sec5_experiments.tex", "sec6_discussion.tex"]:
        raw_tex = (MASTER_DIR / "sections" / sec_name).read_text(encoding="utf-8")
        if sec_name == "sec4_methodology.tex":
            raw_tex = convert_algorithms_to_jcde_env(raw_tex)
        if sec_name == "sec6_discussion.tex":
            # Promote Conclusion to a top-level numbered section (7. Conclusions) per JCDE template
            raw_tex = raw_tex.replace(
                r"\section{Discussion, Computational Scaling, Failure Taxonomy, and Conclusion}",
                r"\section{Discussion, Computational Scaling, and Failure Taxonomy}",
            )
            raw_tex = raw_tex.replace(
                r"\subsection{Conclusion}",
                r"\section{Conclusions}",
            )
        styled_tex = apply_jcde_house_style(raw_tex)
        (JCDE_SEC_DIR / sec_name).write_text(styled_tex, encoding="utf-8")

    # 3. Write JCDE End-Matter, APA 7th References, Appendix A & B (after References), and main.tex
    (JCDE_SEC_DIR / "sec_endmatter.tex").write_text(build_jcde_endmatter_tex(), encoding="utf-8")
    (JCDE_SEC_DIR / "references.tex").write_text(build_jcde_references_tex(), encoding="utf-8")

    sec7_raw = (MASTER_DIR / "sections" / "sec7_appendix.tex").read_text(encoding="utf-8")
    (JCDE_SEC_DIR / "sec7_appendix.tex").write_text(build_jcde_appendix_after_references(sec7_raw), encoding="utf-8")

    (JCDE_TEX_DIR / "main.tex").write_text(build_jcde_main_tex(), encoding="utf-8")

    # 4. Compile manuscript_jcde/main.tex with pdflatex (3 passes)
    for pass_idx in range(1, 4):
        proc = subprocess.run(
            [str(PDFLATEX), "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
            cwd=str(JCDE_TEX_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if proc.returncode != 0:
            print(f"pdflatex failed on pass {pass_idx}:")
            print(proc.stdout[-3000:])
            sys.exit(1)

    pdf_built = JCDE_TEX_DIR / "main.pdf"
    pdf_target = PKG_DIR / "1_Main_Document_JCDE.pdf"
    shutil.copy2(pdf_built, pdf_target)

    # Check log for undefined references or citations
    log_text = (JCDE_TEX_DIR / "main.log").read_text(encoding="utf-8", errors="replace")
    undef_refs = re.findall(r"LaTeX Warning: Reference `[^']+' on page \d+ undefined", log_text)
    undef_cites = re.findall(r"Package natbib Warning: Citation `[^']+' on page \d+ undefined", log_text)
    page_match = re.search(r"Output written on main\.pdf \((\d+) pages, (\d+) bytes\)", log_text)
    pages = int(page_match.group(1)) if page_match else -1
    pdf_bytes = int(page_match.group(2)) if page_match else pdf_target.stat().st_size

    # Verify zero ACIIDS mentions across all JCDE files
    aciids_hits = 0
    for f in JCDE_TEX_DIR.rglob("*.tex"):
        if "aciids" in f.read_text(encoding="utf-8").lower():
            aciids_hits += 1

    # 5. Package 6_Main_Document_Source_JCDE.zip
    zip_src = PKG_DIR / "6_Main_Document_Source_JCDE.zip"
    with zipfile.ZipFile(zip_src, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(JCDE_TEX_DIR / "main.tex", "main.tex")
        for f in sorted(JCDE_SEC_DIR.glob("*.tex")):
            zf.write(f, f"sections/{f.name}")
        for f in sorted(JCDE_FIG_DIR.glob("*.pdf")):
            zf.write(f, f"figures/{f.name}")

    # 6. Package 7_Figures_JCDE.zip
    zip_figs = PKG_DIR / "7_Figures_JCDE.zip"
    with zipfile.ZipFile(zip_figs, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted((MASTER_DIR / "figures").glob("*.*")):
            zf.write(f, f.name)
        for ga_file in ["2_Graphical_Abstract_A4_Portrait.pdf", "2_Graphical_Abstract_HighRes_300dpi.jpg", "2_Graphical_Abstract_Preview.png"]:
            ga_path = PKG_DIR / ga_file
            if ga_path.exists():
                zf.write(ga_path, ga_file)

    # 7. Build github_repo_ready/ and 8_Supplementary_Reproducibility_Code_and_STEP.zip
    GH_DIR.mkdir(parents=True, exist_ok=True)
    for item in GH_DIR.iterdir():
        if item.name == ".git":
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    readme_md = """# NeuroTopo-3D — Official Reproducibility & CAD B-Rep Benchmark Repository

**Paper Title:** *NeuroTopo-3D: Spectral Normal-Flow Manifold Projection for Single-View Polyhedral and CAD B-Rep Reconstruction*  
**Authors:** Tin Nguyen Anh (ORCID: [0009-0009-4454-9541](https://orcid.org/0009-0009-4454-9541)) and Thuyen Phan Thi Le  
**Affiliation:** FPT University, Quy Nhon AI Campus, Vietnam  
**Repository:** [https://github.com/GosuEltin/NeuroTopo-3D](https://github.com/GosuEltin/NeuroTopo-3D)

---

## Overview

This repository contains the reference implementation, symbolic/spectral theorem verification suite, classical non-neural baselines (`Marill-MSDA 1991` and `Lipson–Shpitalni 1996`), multi-seed experimental logs, and exported **ISO 10303-21 (`.STEP` AP214 `MANIFOLD_SOLID_BREP`)** and **Wavefront (`.OBJ`)** 3D CAD models for **NeuroTopo-3D**.

### Key Modules Included
- `src/reasoning/normal_flow_projector_v9.py`: Iterative Normal-Flow Row-Space Levenberg–Marquardt (`NF-RSLM`) with multi-branch Necker reflection continuation and target-free solid-over-dashed ray-crossing disambiguation.
- `src/reasoning/classical_line_drawing_msda_v9.py`: Non-neural classical single-view polyhedral & CAD sketch reconstruction baselines (`Marill-MSDA 1991` and `Lipson–Shpitalni 1996`).
- `src/lifting/q1_lifting_baselines_v9.py`: Permutation-equivariant and occlusion-asymmetric neural 2D-to-3D lifters (`PairGT-Asym`, `PairGT-Sym`, `EGNN`, `GATv2-Edge`, `GraphGPS`).
- `src/cad/step_exporter_v9.py`: Outward 2-manifold B-Rep orientation (half-edge BFS + exact Newell signed-volume test), ISO 10303-21 `.STEP` solid synthesis, CAD kernel sewing tolerance auditor, and Parametric Nullspace CAD Family Explorer.
- `scripts/verify_q1_theorems_v9.py`: Automated `SymPy` symbolic computer algebra and IEEE-754 `float64` verification of Theorems 1–3, Lemma 1, and Proposition 1.
- `scripts/verify_q1_journal_claims_v9.py`: End-to-end 5-gate reproducibility and numerical claim verifier.
- `runs_q1_journal_v9/`: All 10 JSON benchmark reports and 17 exported `.STEP` / `.OBJ` CAD B-Rep solids (`runs_q1_journal_v9/cad_exports/`).

## Quick Verification

```bash
pytest tests/test_q1_theorems_v9.py tests/test_normal_flow_projector_v9.py tests/test_step_exporter_v9.py tests/test_classical_msda_v9.py
python scripts/verify_q1_theorems_v9.py
python scripts/verify_q1_journal_claims_v9.py
```

## License
MIT License.
"""
    (GH_DIR / "README.md").write_text(readme_md, encoding="utf-8")
    (GH_DIR / ".gitignore").write_text("__pycache__/\n*.py[cod]\n.pytest_cache/\n", encoding="utf-8")

    files_to_copy = [
        "src/reasoning/normal_flow_projector_v9.py",
        "src/reasoning/classical_line_drawing_msda_v9.py",
        "src/lifting/q1_lifting_baselines_v9.py",
        "src/cad/step_exporter_v9.py",
    ]
    for s_py in sorted((ROOT / "scripts").glob("*v9.py")):
        files_to_copy.append(f"scripts/{s_py.name}")
    for t_py in sorted((ROOT / "tests").glob("*v9.py")):
        files_to_copy.append(f"tests/{t_py.name}")
    for rel_path in files_to_copy:
        src_p = ROOT / rel_path
        dst_p = GH_DIR / rel_path
        dst_p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_p, dst_p)

    # Copy JSON reports and CAD exports
    gh_runs = GH_DIR / "runs_q1_journal_v9"
    gh_runs.mkdir(parents=True, exist_ok=True)
    for jf in sorted((ROOT / "runs_q1_journal_v9").glob("*.json")):
        shutil.copy2(jf, gh_runs / jf.name)
    gh_cad = gh_runs / "cad_exports"
    gh_cad.mkdir(parents=True, exist_ok=True)
    for cf in sorted((ROOT / "runs_q1_journal_v9" / "cad_exports").glob("*.*")):
        if cf.suffix.lower() in {".step", ".obj"}:
            shutil.copy2(cf, gh_cad / cf.name)

    zip_repro = PKG_DIR / "8_Supplementary_Reproducibility_Code_and_STEP.zip"
    with zipfile.ZipFile(zip_repro, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(GH_DIR.rglob("*")):
            if f.is_file() and ".git" not in f.parts:
                zf.write(f, f.relative_to(GH_DIR).as_posix())

    summary = {
        "jcde_pdf": str(pdf_target),
        "pages": pages,
        "pdf_bytes": pdf_bytes,
        "undef_refs": len(undef_refs),
        "undef_cites": len(undef_cites),
        "aciids_mentions": aciids_hits,
        "zip_source_bytes": zip_src.stat().st_size,
        "zip_figures_bytes": zip_figs.stat().st_size,
        "zip_repro_bytes": zip_repro.stat().st_size,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
