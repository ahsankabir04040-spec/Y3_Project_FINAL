const docxLib = require("docx");
const fss = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType, ImageRun
} = docxLib;

const t = (text, opts = {}) => new TextRun({ text, ...opts });
const tb = (text, opts = {}) => new TextRun({ text, bold: true, ...opts });
const ti = (text, opts = {}) => new TextRun({ text, italics: true, ...opts });
const p = (children, opts = {}) => new Paragraph({ children: Array.isArray(children) ? children : [children], ...opts });

const spc = { before: 100, after: 100, line: 276 };
const eqSpc = { before: 140, after: 140, line: 276 };
const capSpc = { before: 40, after: 160, line: 240 };

// Load real images
const imgDoubleWell = fss.readFileSync("/sessions/wonderful-lucid-euler/mnt/condensate_project/images/double_well.png");
const imgInitCond = fss.readFileSync("/sessions/wonderful-lucid-euler/mnt/condensate_project/images/initial_condition.png");
const imgStencil = fss.readFileSync("/sessions/wonderful-lucid-euler/mnt/condensate_project/images/five_point_stencil.png");
const imgExprVsEP = fss.readFileSync("/sessions/wonderful-lucid-euler/mnt/condensate_project/images/Expression_vs_EP.png");
const imgDistDep = fss.readFileSync("/sessions/wonderful-lucid-euler/mnt/condensate_project/images/Distance_Dependent_Expression.png");
const imgSuperTyp = fss.readFileSync("/sessions/wonderful-lucid-euler/mnt/condensate_project/images/Super_vs_Typical.png");
const imgAnalysis = fss.readFileSync("/sessions/wonderful-lucid-euler/mnt/condensate_project/data/experimental/analysis_summary.png");

function placeholder(label) {
  return [new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 160, after: 40 },
    border: { top: { style: BorderStyle.DASHED, size: 1, color: "AAAAAA", space: 6 }, bottom: { style: BorderStyle.DASHED, size: 1, color: "AAAAAA", space: 6 }, left: { style: BorderStyle.DASHED, size: 1, color: "AAAAAA", space: 6 }, right: { style: BorderStyle.DASHED, size: 1, color: "AAAAAA", space: 6 } },
    children: [t(`[ INSERT ${label} ]`, { bold: true, color: "999999", size: 22 })]
  })];
}

function img(buf, w, h, maxW) {
  maxW = maxW || 5486400;
  const aspect = h / w;
  const rw = Math.round(maxW / 9525);
  const rh = Math.round(maxW * aspect / 9525);
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 160, after: 40 },
    children: [new ImageRun({ data: buf, transformation: { width: rw, height: rh }, type: "png" })]
  });
}

function cap(text) { return p([ti(text, { size: 20 })], { spacing: capSpc, alignment: AlignmentType.CENTER }); }

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Times New Roman", size: 24 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 28, bold: true, font: "Times New Roman" }, paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 26, bold: true, font: "Times New Roman" }, paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 24, bold: true, italics: true, font: "Times New Roman" }, paragraph: { spacing: { before: 160, after: 80 }, outlineLevel: 2 } },
    ]
  },
  numbering: { config: [] },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    children: [

// =====================================================================
//                         3. METHODS
// =====================================================================
new Paragraph({ heading: HeadingLevel.HEADING_1, children: [t("3. Methods")] }),

// --- 3.1 Phase Field Model ---
new Paragraph({ heading: HeadingLevel.HEADING_2, children: [t("3.1 Phase field model")] }),

p([
  t("We model the transcriptional condensate as a binary protein-solvent mixture coupled to a non-conserved RNA field, following the formulation of Goh "),
  ti("et al."),
  t(" [1]. The protein concentration "),
  ti("c"),
  t("("),
  tb("r"),
  t(", "),
  ti("t"),
  t(") evolves by conserved Cahn-Hilliard dynamics; the RNA concentration "),
  ti("m"),
  t("("),
  tb("r"),
  t(", "),
  ti("t"),
  t(") evolves by reaction-diffusion with spatially localised production and first-order degradation. The coupling captures the electrostatic attraction between positively charged IDR proteins and negatively charged nascent RNA transcripts."),
], { spacing: spc }),

// Schematic placeholder
...placeholder("FIGURE 1: Model schematic \u2014 condensate at distance r from promoter, RNA gradient, feedback loop arrows"),
cap("Figure 1. Schematic of the coupled phase field model. A protein condensate (blue circle) is placed at distance r from the promoter (at the origin). The promoter produces RNA (red gradient) in proportion to local protein concentration. The RNA gradient lowers the chemical potential on the near side of the droplet, creating a net thermodynamic force that pulls the condensate toward the source."),

p([t("The total free energy functional is")], { spacing: { before: 100, after: 30 } }),

// Eq 1
p([
  ti("F"),
  t("["),
  ti("c"),
  t(", "),
  ti("m"),
  t("] = \u222B [ (\u03B1/4)("),
  ti("c"),
  t(" \u2013 "),
  ti("c\u0304"),
  t(")\u2074 + (\u03B2/2)("),
  ti("c"),
  t(" \u2013 "),
  ti("c\u0304"),
  t(")\u00B2 + (\u03BA/2)|\u2207"),
  ti("c"),
  t("|\u00B2 + \u03C7"),
  ti("cm"),
  t(" + (\u03B3/2)"),
  ti("c"),
  t("\u00B2"),
  ti("m"),
  t("\u00B2 ] d"),
  tb("r"),
  t("          (1)"),
], { spacing: eqSpc, alignment: AlignmentType.CENTER }),

p([
  t("The first two terms form a double-well potential in "),
  ti("c"),
  t(", centred at the critical concentration "),
  ti("c\u0304"),
  t(" = 4.0 (Figure 2). With \u03B1 = 1.0 and \u03B2 = \u20130.25, the polynomial is binodal: it has two local minima at "),
  ti("c"),
  t("\u207B = "),
  ti("c\u0304"),
  t(" \u2013 \u221A(\u2013\u03B2/\u03B1) = 3.5 (dilute phase) and "),
  ti("c"),
  t("\u207A = "),
  ti("c\u0304"),
  t(" + \u221A(\u2013\u03B2/\u03B1) = 4.5 (dense phase). The system phase-separates because \u03B1 > 0 and \u03B2 < 0; regions with "),
  ti("c"),
  t(" > "),
  ti("c\u0304"),
  t(" evolve toward "),
  ti("c"),
  t("\u207A (inside condensates) and regions with "),
  ti("c"),
  t(" < "),
  ti("c\u0304"),
  t(" evolve toward "),
  ti("c"),
  t("\u207B (outside condensates). The gradient energy (\u03BA = 0.05) penalises abrupt concentration changes, ensuring the interface is smooth with characteristic width \u221A(2\u03BA/|\u03B2|) \u2248 0.63."),
], { spacing: spc }),

// FIGURE 2: Double well (REAL IMAGE)
img(imgDoubleWell, 800, 600),
cap("Figure 2. Double-well free energy landscape f(c) = (\u03B1/4)(c \u2013 c\u0304)\u2074 + (\u03B2/2)(c \u2013 c\u0304)\u00B2. The two minima at c\u207B = 3.5 and c\u207A = 4.5 represent the dilute and dense equilibrium phases. The critical point c\u0304 = 4.0 sits at the local maximum between them."),

p([
  t("The \u03C7 coupling term (\u03C7 = \u20130.1) represents attractive protein-RNA interactions: it lowers the chemical potential of protein in RNA-rich regions, providing the thermodynamic driving force that pulls condensates toward the RNA source. The \u03B3 term models reentrant repulsion at high RNA (charge inversion), but we set \u03B3 = 0 throughout this work since we focus on the low-RNA regime of Figure 1B in [1]."),
], { spacing: spc }),

p([t("Protein evolves by the Cahn-Hilliard equation, which conserves total mass:")], { spacing: { before: 100, after: 30 } }),

// Eq 2
p([
  t("\u2202"),
  ti("c"),
  t("/\u2202"),
  ti("t"),
  t(" = \u2207 \u00B7 ("),
  ti("M"),
  t("\u2080\u2207\u03BC),     \u03BC = \u03B1("),
  ti("c"),
  t(" \u2013 "),
  ti("c\u0304"),
  t(")\u00B3 + \u03B2("),
  ti("c"),
  t(" \u2013 "),
  ti("c\u0304"),
  t(") \u2013 \u03BA\u2207\u00B2"),
  ti("c"),
  t(" + \u03C7"),
  ti("m"),
  t("          (2)"),
], { spacing: eqSpc, alignment: AlignmentType.CENTER }),

p([
  t("The chemical potential \u03BC = \u03B4"),
  ti("F"),
  t("/\u03B4"),
  ti("c"),
  t(" encodes all the forces acting on the protein. The cubic term drives phase separation; the \u2013\u03BA\u2207\u00B2"),
  ti("c"),
  t(" term opposes sharp gradients (surface tension); and the \u03C7"),
  ti("m"),
  t(" term biases protein flux toward high-RNA regions. The protein flux is "),
  tb("J"),
  t(" = \u2013"),
  ti("M"),
  t("\u2080\u2207\u03BC, analogous to Fick\u2019s law but driven by chemical potential gradients rather than concentration gradients. The mobility "),
  ti("M"),
  t("\u2080 = 1.0 is constant."),
], { spacing: spc }),

p([t("RNA obeys a reaction-diffusion equation:")], { spacing: { before: 100, after: 30 } }),

// Eq 3
p([
  t("\u2202"),
  ti("m"),
  t("/\u2202"),
  ti("t"),
  t(" = "),
  ti("D"),
  t("\u2098\u2207\u00B2"),
  ti("m"),
  t(" + "),
  ti("k"),
  t("\u209A exp(\u2013|"),
  tb("r"),
  t("|\u00B2/2\u03C3\u209A\u00B2) \u00B7 "),
  ti("c"),
  t(" \u2013 "),
  ti("k"),
  t("\u2091"),
  ti("m"),
  t("          (3)"),
], { spacing: eqSpc, alignment: AlignmentType.CENTER }),

p([
  t("The three terms represent diffusion ("),
  ti("D"),
  t("\u2098 = 1.0), production at rate "),
  ti("k"),
  t("\u209A near the promoter (Gaussian source with width \u03C3\u209A = 2.5), and uniform degradation ("),
  ti("k"),
  t("\u2091 = 1.0). Production requires both proximity to the promoter and protein presence \u2014 a condensate sitting on the promoter generates far more RNA than one far away. The RNA diffusion length \u2113 = \u221A("),
  ti("D"),
  t("\u2098/"),
  ti("k"),
  t("\u2091) = 1.0 sets how far the gradient extends from the source. This creates the positive feedback loop: RNA attracts the condensate via \u03C7, and the condensate amplifies RNA production when it reaches the promoter."),
], { spacing: spc }),

// --- 3.2 Numerical Implementation ---
new Paragraph({ heading: HeadingLevel.HEADING_2, children: [t("3.2 Numerical implementation")] }),

p([
  t("We solve the coupled system on a 2D Cartesian grid of 256 \u00D7 256 cells covering the domain [\u221225, 25]\u00B2 (\u0394"),
  ti("x"),
  t(" = \u0394"),
  ti("y"),
  t(" \u2248 0.195). The promoter sits at the origin. The initial protein field (Figure 3) is a circular droplet with a tanh profile:"),
], { spacing: spc }),

// Eq 4
p([
  ti("c"),
  t("("),
  tb("r"),
  t(", 0) = ("),
  ti("c"),
  t("\u207A + "),
  ti("c"),
  t("\u207B)/2 + ("),
  ti("c"),
  t("\u207A \u2013 "),
  ti("c"),
  t("\u207B)/2 \u00B7 tanh[("),
  ti("R"),
  t("\u2080 \u2013 |"),
  tb("r"),
  t(" \u2013 "),
  tb("r"),
  t("\u2080|) / "),
  ti("w"),
  t("]          (4)"),
], { spacing: eqSpc, alignment: AlignmentType.CENTER }),

p([
  t("with "),
  ti("c"),
  t("\u207A = 5.5, "),
  ti("R"),
  t("\u2080 = 2.0, "),
  tb("r"),
  t("\u2080 = (10, 0), and "),
  ti("w"),
  t(" = \u221A(2\u03BA/|\u03B2|) \u2248 0.63. The dilute-phase concentration "),
  ti("c"),
  t("\u207B is varied between 3.51 and 3.60 across regimes (Table 1). RNA starts at zero everywhere."),
], { spacing: spc }),

// FIGURE 3: Initial condition (REAL IMAGE)
img(imgInitCond, 800, 800, 4000000),
cap("Figure 3. Initial protein concentration field c(r, 0) on the 256\u00D7256 grid. The circular droplet (c\u207A = 5.5, radius 2) is centred at (10, 0) in a dilute background (c\u207B = 3.53). The promoter at the origin is marked by a green star."),

p([
  t("The discrete Laplacian uses a standard five-point stencil (Figure 4), assembled as a sparse matrix in SciPy\u2019s CSR format. The matrix has dimension 65,536 \u00D7 65,536 but only ~325,000 nonzero entries. No-flux boundary conditions are applied by zeroing fluxes through boundary faces."),
], { spacing: spc }),

// FIGURE 4: Stencil (REAL IMAGE)
img(imgStencil, 600, 750, 3600000),
cap("Figure 4. Five-point finite difference stencil for the 2D Laplacian. The centre cell f\u1D62,\u2C7C and its four neighbours are used to approximate \u2207\u00B2f. The resulting sparse matrix is precomputed once and reused at every time step."),

new Paragraph({ heading: HeadingLevel.HEADING_3, children: [t("Time stepping")] }),

p([
  t("We use a split-step, semi-implicit scheme. RNA is updated first (diffusion implicit, production/decay explicit):"),
], { spacing: spc }),

// Eq 5
p([
  t("("),
  ti("I"),
  t(" \u2013 \u0394"),
  ti("t"),
  ti("D"),
  t("\u2098"),
  tb("L"),
  t(")"),
  ti("m"),
  t("\u207F\u207A\u00B9 = "),
  ti("m"),
  t("\u207F + \u0394"),
  ti("t"),
  t("("),
  ti("k"),
  t("\u209A"),
  ti("S"),
  t("\u2299"),
  ti("c"),
  t("\u207F \u2013 "),
  ti("k"),
  t("\u2091"),
  ti("m"),
  t("\u207F)          (5)"),
], { spacing: eqSpc, alignment: AlignmentType.CENTER }),

p([
  t("Since the left-hand matrix is constant, we precompute its sparse LU factorisation (scipy.sparse.linalg.splu) once and reuse it via forward/back substitution. Protein is then updated explicitly:"),
], { spacing: spc }),

// Eq 6
p([
  ti("c"),
  t("\u207F\u207A\u00B9 = "),
  ti("c"),
  t("\u207F + \u0394"),
  ti("t"),
  ti("M"),
  t("\u2080\u2207\u00B2\u03BC          (6)"),
], { spacing: eqSpc, alignment: AlignmentType.CENTER }),

p([
  t("The explicit Cahn-Hilliard step imposes a stability constraint \u0394"),
  ti("t"),
  t(" < \u0394"),
  ti("x"),
  t("\u2074/(32\u03BA"),
  ti("M"),
  t("\u2080) \u2248 9\u00D710\u207B\u2074; we use \u0394"),
  ti("t"),
  t(" = 5\u00D710\u207B\u2074. Mass conservation was verified to machine precision (~10\u207B\u00B9\u2076 relative error). We also built a 1D radial solver (200 cell-centred finite volumes, cylindrical Laplacian) for rapid parameter sweeps, though the ring geometry introduces an artificial outward drift from curvature asymmetry, so all quantitative results use the 2D solver."),
], { spacing: spc }),

// --- 3.3 Droplet tracking ---
new Paragraph({ heading: HeadingLevel.HEADING_2, children: [t("3.3 Droplet tracking and regime classification")] }),

p([
  t("The droplet is defined as cells where "),
  ti("c"),
  t(" > "),
  ti("c\u0304"),
  t(". Its centre of mass is computed as an excess-concentration-weighted average over dense-phase cells:"),
], { spacing: spc }),

// Eq 7
p([
  tb("r"),
  t("\u209c\u2098 = \u03A3\u1D62 "),
  tb("r"),
  t("\u1D62("),
  ti("c"),
  t("\u1D62 \u2013 "),
  ti("c\u0304"),
  t(")"),
  ti("V"),
  t("\u1D62 / \u03A3\u1D62 ("),
  ti("c"),
  t("\u1D62 \u2013 "),
  ti("c\u0304"),
  t(")"),
  ti("V"),
  t("\u1D62          (7)"),
], { spacing: eqSpc, alignment: AlignmentType.CENTER }),

p([
  t("The aspect ratio is extracted from the inertia tensor eigenvalues to detect elongation. Simulations are classified into four regimes using time-series criteria: Regime I (dissolution) if the radius drops below 20% of its initial value; Regime II (renucleation) if the droplet dissolves but a new one forms near the promoter; Regime III (directed motion) if the droplet persists and moves at least 0.5 units inward with >40% of steps showing inward displacement; Regime IV (elongation) if Regime III criteria are met and the radius grows by >20%."),
], { spacing: spc }),

// --- 3.4 Analytical velocity ---
new Paragraph({ heading: HeadingLevel.HEADING_2, children: [t("3.4 Analytical droplet velocity")] }),

p([
  t("In the sharp-interface limit, the droplet velocity can be derived analytically [1, Eq. 11]. The RNA gradient creates an asymmetric chemical potential across the droplet: the side facing the promoter sees higher RNA, so \u03BC is lower there, pulling protein inward. The velocity in "),
  ti("d"),
  t(" dimensions is:"),
], { spacing: spc }),

// Eq 8
p([
  ti("v"),
  t(" = \u2013("),
  ti("M"),
  t("\u2080\u03C7"),
  ti("d"),
  t(" / \u0394"),
  ti("cV"),
  t(") \u222B\u2091 \u2207"),
  ti("m"),
  t(" \u00B7 "),
  tb("e\u0302"),
  t("\u1D65 d\u1D48"),
  tb("r"),
  t("          (8)"),
], { spacing: eqSpc, alignment: AlignmentType.CENTER }),

p([
  t("where \u0394"),
  ti("c"),
  t(" = "),
  ti("c"),
  t("\u207A \u2013 "),
  ti("c"),
  t("\u207B and "),
  ti("V"),
  t(" is droplet volume. The velocity is proportional to "),
  ti("M"),
  t("\u2080 (mobility), \u03C7 (attraction strength), and the RNA gradient asymmetry, and inversely proportional to \u0394"),
  ti("c"),
  t(" (a heavier, richer droplet is harder to move). It peaks at "),
  ti("r"),
  t(" \u2248 "),
  ti("R"),
  t(" and drops to zero at "),
  ti("r"),
  t(" = 0 (symmetric gradient when the droplet sits on the source). We evaluate this numerically using SciPy\u2019s Gaussian quadrature."),
], { spacing: spc }),

// --- 3.5 Rouse polymer ---
new Paragraph({ heading: HeadingLevel.HEADING_2, children: [t("3.5 Rouse polymer model")] }),

p([
  t("Following Section V of [1], we model the chromatin between enhancer and promoter as a Rouse chain: "),
  ti("N"),
  t(" beads linked by harmonic springs, each undergoing overdamped Brownian dynamics. The enhancer is pinned at one end; a harmonic trap at the other pulls the promoter bead toward the condensate position. The equation of motion for bead "),
  ti("i"),
  t(" is:"),
], { spacing: spc }),

// Eq 9
p([
  t("\u03B6 d"),
  ti("x"),
  t("\u1D62/d"),
  ti("t"),
  t(" = "),
  ti("k"),
  t("\u209B("),
  ti("x"),
  t("\u1D62\u208A\u2081 + "),
  ti("x"),
  t("\u1D62\u208B\u2081 \u2013 2"),
  ti("x"),
  t("\u1D62) + "),
  ti("F"),
  t("\u1D62 + \u03B6\u221A(2"),
  ti("D"),
  t("/\u0394"),
  ti("t"),
  t(")\u03B7\u1D62          (9)"),
], { spacing: eqSpc, alignment: AlignmentType.CENTER }),

p([
  t("where "),
  ti("k"),
  t("\u209B is the spring constant, \u03B6 the drag coefficient, "),
  ti("D"),
  t(" = "),
  ti("k"),
  t("\u0042"),
  ti("T"),
  t("/\u03B6 the thermal diffusivity, and \u03B7\u1D62 is unit Gaussian noise. The spring term is a discrete Laplacian encoding Hookean forces between neighbours. The external force "),
  ti("F"),
  t("\u1D62 = \u2013"),
  ti("k"),
  t("\u209C\u2090\u209A("),
  ti("x"),
  t("\u209A \u2013 "),
  ti("x"),
  t("\u209C\u2092\u2099\u2091)\u03B4\u1D62,\u2099 acts only on the promoter bead. We run 10\u2075 steps, discard the first half for equilibration, and count contacts (enhancer-promoter distance < 0.5) to compute contact probabilities. Sweeping genomic distance (10\u2013150 beads) and trap strength produces the heatmaps presented in the Results."),
], { spacing: spc }),

// --- 3.6 Experimental data ---
new Paragraph({ heading: HeadingLevel.HEADING_2, children: [t("3.6 Experimental data")] }),

p([
  t("We compared model predictions with mESC Micro-C data from Hsieh "),
  ti("et al."),
  t(" [2] and matched RNA-seq expression data. Enhancer-promoter contact frequencies were normalised by expected counts at each genomic distance to remove the baseline distance-decay. Expression levels from RNA-seq were assigned to promoters via gene annotation. Super-enhancers were identified by H3K27ac ChIP-seq signal strength."),
], { spacing: spc }),

// --- 3.7 Parameter table ---
new Paragraph({ heading: HeadingLevel.HEADING_2, children: [t("3.7 Parameters")] }),

// Table
(() => {
  const border = { style: BorderStyle.SINGLE, size: 1, color: "888888" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const cm = { top: 50, bottom: 50, left: 80, right: 80 };
  const hdr = { fill: "D9E2F3", type: ShadingType.CLEAR };
  const hc = (txt, w) => new TableCell({ borders, width: { size: w, type: WidthType.DXA }, shading: hdr, margins: cm, children: [p([tb(txt, { size: 20 })], { alignment: AlignmentType.CENTER })] });
  const c = (txt, w) => new TableCell({ borders, width: { size: w, type: WidthType.DXA }, margins: cm, children: [p([t(txt, { size: 20 })], { alignment: AlignmentType.CENTER })] });
  return new Table({
    width: { size: 9360, type: WidthType.DXA }, columnWidths: [1800, 2520, 2520, 2520],
    rows: [
      new TableRow({ children: [hc("Regime", 1800), hc("k\u209A", 2520), hc("c\u207B", 2520), hc("Behaviour", 2520)] }),
      new TableRow({ children: [c("I", 1800), c("0.05", 2520), c("3.51", 2520), c("Dissolution", 2520)] }),
      new TableRow({ children: [c("II", 1800), c("0.40", 2520), c("3.51", 2520), c("Renucleation", 2520)] }),
      new TableRow({ children: [c("III", 1800), c("0.08", 2520), c("3.53", 2520), c("Directed motion", 2520)] }),
      new TableRow({ children: [c("IV", 1800), c("0.25", 2520), c("3.60", 2520), c("Elongation", 2520)] }),
    ]
  });
})(),

p([
  ti("Table 1. Regime-specific parameters. Fixed: \u03B1 = 1, \u03B2 = \u20130.25, \u03BA = 0.05, c\u0304 = 4.0, \u03C7 = \u20130.1, \u03B3 = 0, M\u2080 = 1, D\u2098 = 1, k\u2091 = 1, \u03C3\u209A = 2.5, c\u207A(0) = 5.5, R\u2080 = 2, r\u2080 = 10. Grid: 256\u00B2 on [\u221225,25]\u00B2, \u0394t = 5\u00D710\u207B\u2074."),
], { spacing: { before: 40, after: 100 }, size: 20 }),

// --- 3.8 Software ---
new Paragraph({ heading: HeadingLevel.HEADING_2, children: [t("3.8 Software")] }),

p([
  t("All code was written in Python 3 (NumPy, SciPy, Matplotlib), organised into modules for physics, numerics, solvers, and analysis. Parameters are stored as serialisable dataclasses with factory methods for each regime. A 2D simulation of 500 time units runs in ~30 minutes on a single CPU core."),
], { spacing: spc }),


// =====================================================================
//                         4. RESULTS
// =====================================================================
new Paragraph({ heading: HeadingLevel.HEADING_1, children: [t("4. Results")] }),

// --- 4.1 Phase diagram and regime behaviour ---
new Paragraph({ heading: HeadingLevel.HEADING_2, children: [t("4.1 Four dynamical regimes")] }),

p([
  t("Simulating across the ("),
  ti("k"),
  t("\u209A, "),
  ti("c"),
  t("\u207B) parameter plane reproduces the four dynamical regimes identified by Goh "),
  ti("et al."),
  t(" [1]. Figure 5 shows representative 2D snapshots of the protein and RNA fields for each regime."),
], { spacing: spc }),

...placeholder("FIGURE 5: 2D simulation snapshots for Regimes I\u2013IV \u2014 protein (top) and RNA (bottom) at selected time points"),
cap("Figure 5. Snapshots of the four dynamical regimes from 2D simulations. Top: protein concentration c(r,t). Bottom: RNA concentration m(r,t). I: dissolution. II: renucleation at the promoter. III: directed motion toward the promoter. IV: elongated inward motion."),

p([
  t("The phase diagram (Figure 6) maps the regime boundaries in parameter space. At low "),
  ti("c"),
  t("\u207B (near the lower binodal), there is barely enough protein to sustain a condensate; even weak RNA redistribution dissolves it (Regime I). At the same "),
  ti("c"),
  t("\u207B but high "),
  ti("k"),
  t("\u209A, the RNA gradient nucleates a new condensate directly at the promoter (Regime II). At higher "),
  ti("c"),
  t("\u207B the droplet is thermodynamically stable and the RNA gradient pulls it inward (III), with elongation appearing at still higher "),
  ti("c"),
  t("\u207B (IV)."),
], { spacing: spc }),

...placeholder("FIGURE 6: Phase diagram in (k\u209A, c\u207B) plane, coloured by regime"),
cap("Figure 6. Phase diagram in the (k\u209A, c\u207B) parameter plane. Each point is coloured by its classified regime, reproducing the structure of Figure 1B in [1]."),

// --- 4.2 Droplet trajectories ---
new Paragraph({ heading: HeadingLevel.HEADING_2, children: [t("4.2 Droplet trajectories")] }),

p([
  t("Figure 7 tracks the droplet centre-of-mass distance from the promoter over time for each regime."),
], { spacing: spc }),

...placeholder("FIGURE 7: Droplet distance from promoter vs. time for all four regimes"),
cap("Figure 7. Droplet centre-of-mass distance from the promoter over time. Regime I: rapid dissolution. Regime II: dissolution followed by renucleation near origin. Regime III: steady inward drift. Regime IV: inward drift with increasing aspect ratio."),

// --- 4.3 Velocity validation ---
new Paragraph({ heading: HeadingLevel.HEADING_2, children: [t("4.3 Velocity profile")] }),

p([
  t("Figure 8 compares the analytical velocity (Eq. 8) with velocities measured from 2D simulations."),
], { spacing: spc }),

...placeholder("FIGURE 8: Analytical and simulated velocity vs. distance from promoter"),
cap("Figure 8. Droplet velocity as a function of distance from the promoter. Solid: analytical prediction (Eq. 8). Points: velocities from 2D simulation. The non-monotonic profile peaks near r \u2248 R \u2014 velocity is maximised when the leading edge just contacts the RNA source."),

// --- 4.4 Enhancer-promoter contacts ---
new Paragraph({ heading: HeadingLevel.HEADING_2, children: [t("4.4 Enhancer-promoter contact probability")] }),

p([
  t("The Rouse polymer simulations produce a contact probability heatmap (Figure 9) as a function of genomic distance and condensate trap strength."),
], { spacing: spc }),

...placeholder("FIGURE 9: Contact probability heatmap (YlOrRd, genomic distance vs trap strength)"),
cap("Figure 9. Contact probability heatmap from Rouse polymer Brownian dynamics. Contact probability increases with trap strength (proportional to k\u209A) and decreases with genomic distance, consistent with condensate-mediated looping being strongest for highly transcribed genes at intermediate separations."),

// --- 4.5 Experimental validation ---
new Paragraph({ heading: HeadingLevel.HEADING_2, children: [t("4.5 Experimental comparison")] }),

p([
  t("The model\u2019s predictions are tested against mESC chromatin data. Figure 10 shows a positive correlation between gene expression and enhancer-promoter contact frequency."),
], { spacing: spc }),

img(imgExprVsEP, 640, 480),
cap("Figure 10. Expression (log\u2082(TPM+1)) vs. enhancer-promoter contact frequency (log O/E) for mESC pairs. Spearman \u03C1 = 0.372. Higher expression correlates with elevated contact, consistent with the model."),

p([
  t("Stratifying by genomic distance (Figure 11) reveals that the expression-contact correlation peaks at 75\u2013150 kb, matching the model\u2019s prediction that condensate-mediated looping is most effective at intermediate separations."),
], { spacing: spc }),

img(imgDistDep, 1000, 500),
cap("Figure 11. Distance-dependent Spearman correlation between expression and contact frequency. The correlation peaks at intermediate distances (75\u2013150 kb) where the RNA gradient can bridge the gap but polymer entropy cost is not yet prohibitive."),

p([
  t("Super-enhancers show a stronger expression-contact correlation than typical enhancers (Figure 12), consistent with larger condensates and steeper RNA gradients at multi-gene transcriptional hubs."),
], { spacing: spc }),

img(imgSuperTyp, 1000, 400),
cap("Figure 12. Expression vs. contact for super-enhancers (\u03C1 = 0.389, left) and typical enhancers (\u03C1 = 0.308, right). The stronger correlation at super-enhancers supports the condensate hub model."),

img(imgAnalysis, 800, 800),
cap("Figure 13. Summary of experimental analysis. (A) Expression vs. E-P contacts. (B) E-P distance distribution. (C) Distance-stratified correlation (orange: significant at p < 0.05). (D) Expression distribution."),

    ]
  }]
});

Packer.toBuffer(doc).then(buffer => {
  fss.writeFileSync("/sessions/wonderful-lucid-euler/mnt/condensate_project/methods_section.docx", buffer);
  console.log("Done.");
});
