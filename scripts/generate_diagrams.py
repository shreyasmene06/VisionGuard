#!/usr/bin/env python3
"""VisionGuard: System Architecture and UML Diagram Generator.

Generates the 5 required system design and architectural diagrams:
1. docs/architecture.png  - High-level layered system architecture
2. docs/workflow.png      - End-to-end data processing workflow
3. docs/use_case.png      - System actors and use-case interactions
4. docs/sequence.png      - Chronological object invocation sequence
5. docs/component.png     - UML Component and module dependency layout
"""

from pathlib import Path
import subprocess
import sys


def generate_architecture_dot() -> str:
    return """
digraph Architecture {
    graph [rankdir=TB, bgcolor="#0f141c", fontname="Helvetica", pad="0.4", nodesep="0.6", ranksep="0.7"];
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=11, fontcolor="#ffffff", penwidth=1.5];
    edge [fontname="Helvetica", fontsize=9, fontcolor="#a0aec0", color="#4a5568", arrowsize=0.8, penwidth=1.2];

    subgraph cluster_input {
        label = "Layer 1: Input & Ingestion";
        fontcolor = "#63b3ed";
        color = "#2b6cb0";
        style = "dashed,rounded";
        bgcolor = "#171f2c";

        inp_single [label="Single Road Image\\n(Dashcam / Mobile RGB)", fillcolor="#1a365d", color="#3182ce"];
        inp_batch [label="Batch Image Directory\\n(Survey Corridor Frames)", fillcolor="#1a365d", color="#3182ce"];
        inp_cli [label="CLI Argument Parser\\n(--input, --model, --conf)", fillcolor="#2a4365", color="#4299e1"];
    }

    subgraph cluster_preprocessing {
        label = "Layer 2: Preprocessing & Normalization";
        fontcolor = "#4fd1c5";
        color = "#285e61";
        style = "dashed,rounded";
        bgcolor = "#142429";

        prep_val [label="Input Integrity Validation\\n(Min dims, corrupt check)", fillcolor="#1d4044", color="#319795"];
        prep_clahe [label="Adaptive Contrast (CLAHE)\\n(L-channel equalization)", fillcolor="#1d4044", color="#38b2ac"];
        prep_denoise [label="Bilateral Edge-Preserving Filter\\n(Asphalt grain suppression)", fillcolor="#1d4044", color="#38b2ac"];
        prep_roi [label="Drivable ROI Masking\\n(Horizon / sky exclusion)", fillcolor="#1d4044", color="#4fd1c5"];
    }

    subgraph cluster_detection {
        label = "Layer 3: Multi-Mode Hazard Detection";
        fontcolor = "#f6ad55";
        color = "#7b341e";
        style = "dashed,rounded";
        bgcolor = "#261a1a";

        det_yolo [label="Deep Learning Engine\\n(YOLOv8 Feature Pyramid + Head)", fillcolor="#742a2a", color="#e53e3e"];
        det_cv [label="Classical CV Fallback\\n(Black-Hat + Directional Kernels)", fillcolor="#7b341e", color="#dd6b20"];
        det_nms [label="Non-Maximum Suppression (NMS)\\n(IoU Overlap Suppression)", fillcolor="#652b19", color="#ed8936"];
    }

    subgraph cluster_analytics {
        label = "Layer 4: Geometric Analysis & Severity Quantification";
        fontcolor = "#b794f4";
        color = "#553c9a";
        style = "dashed,rounded";
        bgcolor = "#1f1a2e";

        sev_geom [label="Geometric Descriptors\\n(Area px, Relative %, Aspect Ratio)", fillcolor="#44337a", color="#805ad5"];
        sev_prox [label="Perspective Proximity\\n(Near-field Wheel Path Weight)", fillcolor="#44337a", color="#9f7aea"];
        sev_calc [label="Severity Index Formulation\\n(0-100 Score & Tier Mapping)", fillcolor="#553c9a", color="#b794f4"];
        pci_calc [label="ASTM D6433 Pavement Condition\\n(PCI Deduct Aggregation)", fillcolor="#553c9a", color="#d6bcfa"];
    }

    subgraph cluster_output {
        label = "Layer 5: Visualization & Headless Reporting";
        fontcolor = "#68d391";
        color = "#22543d";
        style = "dashed,rounded";
        bgcolor = "#13231b";

        vis_annot [label="HUD Annotated Image\\n(Corner brackets, pills, telemetry)", fillcolor="#1c4532", color="#38a169"];
        rep_json [label="Structured JSON Telemetry\\n(Inspection results & coordinates)", fillcolor="#1c4532", color="#48bb78"];
        rep_csv [label="Engineering CSV Logs\\n(Detections & PCI summaries)", fillcolor="#1c4532", color="#68d391"];
    }

    // Connections
    inp_single -> inp_cli;
    inp_batch -> inp_cli;
    inp_cli -> prep_val;
    prep_val -> prep_clahe;
    prep_clahe -> prep_denoise;
    prep_denoise -> prep_roi;

    prep_roi -> det_yolo [label="RGB Tensor"];
    prep_roi -> det_cv [label="Grayscale/Morph"];
    det_yolo -> det_nms;
    det_cv -> det_nms;

    det_nms -> sev_geom;
    sev_geom -> sev_prox;
    sev_prox -> sev_calc;
    sev_calc -> pci_calc;

    pci_calc -> vis_annot;
    pci_calc -> rep_json;
    pci_calc -> rep_csv;
}
"""


def generate_workflow_dot() -> str:
    return """
digraph Workflow {
    graph [rankdir=LR, bgcolor="#0f141c", fontname="Helvetica", pad="0.4", nodesep="0.5", ranksep="0.6"];
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=11, fontcolor="#ffffff", penwidth=1.5];
    edge [fontname="Helvetica", fontsize=9, fontcolor="#a0aec0", color="#4a5568", arrowsize=0.8, penwidth=1.2];

    start [shape=ellipse, label="Raw Image Ingestion", fillcolor="#2b6cb0", color="#63b3ed"];
    val [label="Integrity Check\\n& Color Standardize", fillcolor="#1a365d", color="#3182ce"];
    clahe [label="CLAHE Illumination\\nBalancing", fillcolor="#1d4044", color="#38b2ac"];
    denoise [label="Bilateral Texture\\nDenoising", fillcolor="#1d4044", color="#38b2ac"];
    detect [label="Multi-Class Hazard\\nDetection", fillcolor="#7b341e", color="#ed8936"];
    filter_nms [label="Confidence Filter\\n& NMS Merging", fillcolor="#652b19", color="#dd6b20"];
    extract_geom [label="Geometric & Proximity\\nExtraction", fillcolor="#44337a", color="#9f7aea"];
    calc_sev [label="Mathematical\\nSeverity Scoring", fillcolor="#553c9a", color="#b794f4"];
    pci_deduct [label="ASTM D6433 PCI\\nState Calculation", fillcolor="#4c1d95", color="#c084fc"];
    render [label="Headless HUD\\nCanvas Rendering", fillcolor="#1c4532", color="#48bb78"];
    export [label="JSON / CSV / JPG\\nPersistence", fillcolor="#1c4532", color="#68d391"];
    done [shape=ellipse, label="Inspection Complete", fillcolor="#22543d", color="#68d391"];

    start -> val -> clahe -> denoise -> detect -> filter_nms -> extract_geom -> calc_sev -> pci_deduct -> render -> export -> done;
}
"""


def generate_use_case_dot() -> str:
    return """
digraph UseCase {
    graph [rankdir=LR, bgcolor="#0f141c", fontname="Helvetica", pad="0.4", nodesep="0.5", ranksep="0.8"];
    node [fontname="Helvetica", fontsize=11, fontcolor="#ffffff", penwidth=1.5];
    edge [fontname="Helvetica", fontsize=9, fontcolor="#a0aec0", color="#4a5568", arrowsize=0.8, penwidth=1.2];

    actor_engineer [shape=oval, label="Municipal Road\\nEngineer / Inspector", fillcolor="#1a365d", style="filled", color="#4299e1"];
    actor_evaluator [shape=oval, label="Academic Automated\\nCourse Evaluator", fillcolor="#2c5282", style="filled", color="#63b3ed"];
    actor_ops [shape=oval, label="Highway Maintenance\\nRepair Crew", fillcolor="#2b6cb0", style="filled", color="#90cdf4"];

    subgraph cluster_boundary {
        label = "VisionGuard System Boundary";
        fontcolor = "#e2e8f0";
        color = "#4a5568";
        style = "dashed,rounded";
        bgcolor = "#171f2c";
        node [shape=ellipse, style="filled", fillcolor="#2d3748", color="#718096"];

        uc_cli [label="Execute Headless CLI\\n(--input, --output)", color="#4fd1c5"];
        uc_pothole [label="Detect Road Hazards\\n(Potholes, Cracks, Debris)", color="#f6ad55"];
        uc_severity [label="Compute Deterministic\\nSeverity Scores", color="#b794f4"];
        uc_pci [label="Calculate Pavement\\nCondition Index (PCI)", color="#d6bcfa"];
        uc_hud [label="Generate Color-Coded\\nHUD Telemetry Visuals", color="#68d391"];
        uc_export [label="Export Machine-Readable\\nJSON & CSV Logs", color="#68d391"];
        uc_eval [label="Run Precision/Recall\\n& mAP Evaluation", color="#f687b3"];
    }

    actor_engineer -> uc_cli;
    actor_engineer -> uc_pci;
    actor_engineer -> uc_hud;

    actor_evaluator -> uc_cli;
    actor_evaluator -> uc_eval;
    actor_evaluator -> uc_export;

    actor_ops -> uc_severity;
    actor_ops -> uc_hud;

    uc_cli -> uc_pothole [label="<<include>>", style="dashed"];
    uc_pothole -> uc_severity [label="<<include>>", style="dashed"];
    uc_severity -> uc_pci [label="<<include>>", style="dashed"];
    uc_pci -> uc_hud [label="<<include>>", style="dashed"];
    uc_pci -> uc_export [label="<<include>>", style="dashed"];
}
"""


def generate_sequence_dot() -> str:
    return """
digraph Sequence {
    graph [rankdir=TB, bgcolor="#0f141c", fontname="Helvetica", pad="0.4", nodesep="0.6", ranksep="0.5"];
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=11, fontcolor="#ffffff", penwidth=1.5];
    edge [fontname="Helvetica", fontsize=9, fontcolor="#a0aec0", color="#4a5568", arrowsize=0.8, penwidth=1.2];

    cli [label="CLI / User\\n(run_pipeline.py)", fillcolor="#1a365d", color="#3182ce"];
    pipe [label="VisionGuardPipeline", fillcolor="#2b6cb0", color="#4299e1"];
    prep [label="ImagePreprocessor", fillcolor="#1d4044", color="#38b2ac"];
    det [label="HazardDetector\\n(YOLO / Classical)", fillcolor="#7b341e", color="#ed8936"];
    sev [label="SeverityEngine", fillcolor="#44337a", color="#9f7aea"];
    ana [label="RoadAnalyzer", fillcolor="#553c9a", color="#b794f4"];
    vis [label="RoadVisualizer", fillcolor="#1c4532", color="#48bb78"];

    cli -> pipe [label="1. process_image(path, opts)"];
    pipe -> prep [label="2. preprocess(raw_img)"];
    prep -> pipe [label="3. PreprocessResult (CLAHE + Denoise)"];
    pipe -> det [label="4. detect(processed_img)"];
    det -> pipe [label="5. DetectionResult (boxes, classes, conf)"];
    pipe -> sev [label="6. assess_all(detections, img_shape)"];
    sev -> pipe [label="7. List[HazardSeverityAssessment]"];
    pipe -> ana [label="8. analyze(assessments)"];
    ana -> pipe [label="9. RoadConditionAnalysis (PCI, state)"];
    pipe -> vis [label="10. annotate(img, analysis)"];
    vis -> pipe [label="11. Saved annotated image"];
    pipe -> cli [label="12. PipelineResult (JSON/CSV written)"];
}
"""


def generate_component_dot() -> str:
    return """
digraph Component {
    graph [rankdir=TB, bgcolor="#0f141c", fontname="Helvetica", pad="0.4", nodesep="0.6", ranksep="0.7"];
    node [shape=component, style="filled", fontname="Helvetica", fontsize=11, fontcolor="#ffffff", penwidth=1.5];
    edge [fontname="Helvetica", fontsize=9, fontcolor="#a0aec0", color="#4a5568", arrowsize=0.8, penwidth=1.2];

    subgraph cluster_core {
        label = "Core Package: src/";
        fontcolor = "#63b3ed";
        color = "#2b6cb0";
        style = "dashed,rounded";
        bgcolor = "#171f2c";

        c_config [label="config.py\\n(Enums, Defaults, Weights)", fillcolor="#1a365d", color="#3182ce"];
        c_prep [label="preprocessing.py\\n(ImagePreprocessor)", fillcolor="#1d4044", color="#38b2ac"];
        c_det [label="detector.py\\n(BaseDetector, YOLO, Classical)", fillcolor="#7b341e", color="#ed8936"];
        c_sev [label="severity.py\\n(SeverityEngine)", fillcolor="#44337a", color="#9f7aea"];
        c_ana [label="analyzer.py\\n(RoadAnalyzer, PCI)", fillcolor="#553c9a", color="#b794f4"];
        c_vis [label="visualization.py\\n(RoadVisualizer)", fillcolor="#1c4532", color="#48bb78"];
        c_met [label="metrics.py\\n(evaluate_detections, mAP)", fillcolor="#702459", color="#ed64a6"];
        c_pipe [label="pipeline.py\\n(VisionGuardPipeline)", fillcolor="#2c5282", color="#63b3ed"];
    }

    subgraph cluster_scripts {
        label = "CLI Scripts: scripts/";
        fontcolor = "#f6ad55";
        color = "#7b341e";
        style = "dashed,rounded";
        bgcolor = "#261a1a";

        s_run [label="run_pipeline.py\\n(Main Entry Point)", fillcolor="#742a2a", color="#e53e3e"];
        s_eval [label="evaluate.py\\n(Benchmark Runner)", fillcolor="#742a2a", color="#e53e3e"];
    }

    subgraph cluster_external {
        label = "External CV & DL Libraries";
        fontcolor = "#cbd5e0";
        color = "#4a5568";
        style = "dashed,rounded";
        bgcolor = "#1a202c";
        node [shape=box, style="filled,rounded", fillcolor="#2d3748", color="#718096"];

        ext_cv [label="OpenCV\\n(cv2 headless)"];
        ext_np [label="NumPy"];
        ext_torch [label="PyTorch / Ultralytics"];
    }

    // Dependencies
    s_run -> c_pipe;
    s_eval -> c_pipe;
    s_eval -> c_met;

    c_pipe -> c_prep;
    c_pipe -> c_det;
    c_pipe -> c_sev;
    c_pipe -> c_ana;
    c_pipe -> c_vis;
    c_pipe -> c_config;

    c_prep -> ext_cv;
    c_prep -> ext_np;
    c_det -> ext_cv;
    c_det -> ext_torch;
    c_vis -> ext_cv;
    c_sev -> c_config;
    c_ana -> c_config;
}
"""


def compile_dot(dot_source: str, output_png: Path) -> None:
    """Compile dot code into high-resolution 300-DPI PNG image."""
    output_png.parent.mkdir(parents=True, exist_ok=True)
    dot_file = output_png.with_suffix(".dot")

    with open(dot_file, "w", encoding="utf-8") as f:
        f.write(dot_source)

    try:
        subprocess.run(
            ["dot", "-Tpng", "-Gdpi=200", str(dot_file), "-o", str(output_png)],
            check=True,
            capture_output=True,
        )
        print(f"[OK] Generated diagram: {output_png}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to compile {dot_file}: {e.stderr.decode()}", file=sys.stderr)
        raise


def main() -> int:
    docs_dir = Path(__file__).resolve().parent.parent / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    diagrams = [
        ("architecture.png", generate_architecture_dot()),
        ("workflow.png", generate_workflow_dot()),
        ("use_case.png", generate_use_case_dot()),
        ("sequence.png", generate_sequence_dot()),
        ("component.png", generate_component_dot()),
    ]

    print("=" * 78)
    print("  Generating VisionGuard Architectural & Design Diagrams")
    print("=" * 78)

    for filename, dot_src in diagrams:
        out_path = docs_dir / filename
        compile_dot(dot_src, out_path)

    print("\n[SUCCESS] All 5 system diagrams successfully compiled to docs/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
