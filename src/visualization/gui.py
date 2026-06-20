import sys
import os
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import gradio as gr
import pandas as pd

from visualization.core import run_cam_pipeline, CAM_METHODS


def safe_int_or_none(value):
    if value is None:
        return None
    try:
        if math.isnan(value):
            return None
    except (TypeError, ValueError):
        pass
    return int(value)


def _single_file_path(file_value):
    if file_value is None:
        return None
    return str(file_value)


def _paths_to_root_dir(file_list):
    if not file_list:
        return None
    str_paths = [os.path.abspath(str(p)) for p in file_list]
    common = os.path.commonpath(str_paths)
    if os.path.isfile(common):
        return os.path.dirname(common)
    return common


_FILE_ROW_HEIGHT = 76


def generate_cams(
    uploaded_files,
    folder_dir_files,
    dataset_dir_files,
    dataset_split, sample_n, sample_classes, sample_seed,
    network, weights_file, num_classes, uses_ef_modules,
    cam_method, target_class, device, output_dir_text,
):
    try:
        target_class = safe_int_or_none(target_class)
        sample_seed = safe_int_or_none(sample_seed)

        use_upload = uploaded_files is not None and len(uploaded_files) > 0
        use_folder = not use_upload and folder_dir_files is not None and len(folder_dir_files) > 0
        use_sample = not use_upload and not use_folder

        input_paths = None
        input_dir = None

        if use_upload:
            input_paths = [
                f.name if hasattr(f, "name") else str(f)
                for f in uploaded_files
            ][:16]
        elif use_folder:
            input_dir = _paths_to_root_dir(folder_dir_files)

        weights_path = _single_file_path(weights_file)
        if not weights_path:
            raise gr.Error("Please select a weights file.")

        dataset_path = None
        if use_sample:
            dataset_path = _paths_to_root_dir(dataset_dir_files)
            if not dataset_path:
                raise gr.Error("Please select the AffectNet dataset root folder.")

        out = (output_dir_text or "").strip()
        output_dir = str(Path(out).expanduser()) if out else "./cam_output"

        results = run_cam_pipeline(
            network=network,
            weights_path=weights_path,
            num_classes=int(num_classes),
            uses_ef_modules=uses_ef_modules,
            input_paths=input_paths,
            input_dir=input_dir,
            sample=use_sample,
            dataset_path=dataset_path,
            sample_split=dataset_split if use_sample else "test",
            sample_n=int(sample_n) if use_sample else 1,
            sample_classes=int(sample_classes) if use_sample else 7,
            sample_seed=sample_seed if use_sample else None,
            cam_method=cam_method,
            target_class=target_class,
            output_dir=output_dir,
            device=device,
        )

        gallery = []
        table_rows = []
        for r in results:
            caption = f"{Path(r['input_path']).name} → {r['predicted_name']} ({r['confidence']:.1%})"
            if r["gt_name"] is not None:
                caption += f" [GT: {r['gt_name']}]"
            gallery.append((r["cam_image"], caption))
            table_rows.append({
                "File": Path(r["input_path"]).name,
                "Predicted": r["predicted_name"],
                "Confidence": f"{r['confidence']:.1%}",
                "Ground Truth": r["gt_name"] or "",
                "Output Path": r["output_path"],
            })

        df = pd.DataFrame(table_rows)
        return gallery, df

    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(str(e))


def main():
    default_device = "cuda" if torch.cuda.is_available() else "cpu"

    with gr.Blocks(title="CAM Visualization Tool") as demo:
        gr.Markdown("# CAM Visualization Tool")

        with gr.Tabs():
            with gr.Tab("Upload Images"):
                uploaded_files = gr.File(
                    file_count="multiple", file_types=["image"],
                    label="Upload images (max 16)",
                    height=_FILE_ROW_HEIGHT,
                )
            with gr.Tab("Folder Path"):
                folder_dir_files = gr.File(
                    file_count="directory",
                    label="Select folder containing images",
                    height=_FILE_ROW_HEIGHT,
                )
            with gr.Tab("Random Sample"):
                dataset_dir_files = gr.File(
                    file_count="directory",
                    label="Select AffectNet root folder",
                    height=_FILE_ROW_HEIGHT,
                )
                dataset_split = gr.Radio(
                    choices=["train", "test"], value="test", label="Dataset split",
                )
                sample_n = gr.Slider(
                    minimum=1, maximum=16, step=1, value=4, label="Number of images",
                )
                sample_classes = gr.Radio(
                    choices=[7, 8], value=7,
                    label="Class subset (7=folders 0-6, 8=folders 0-7)",
                    type="value",
                )
                sample_seed = gr.Number(label="Seed (optional)", precision=0)

        with gr.Row(equal_height=True):
            network = gr.Dropdown(
                choices=["EF", "MLDG"], value="EF", label="Network",
            )
            weights_file = gr.File(
                file_count="single",
                file_types=[".pth", ".tar"],
                label="Weights file",
                height=_FILE_ROW_HEIGHT,
            )
            num_classes = gr.Radio(
                choices=[7, 8], value=7, label="Num classes", type="value",
            )
            uses_ef_modules = gr.Checkbox(value=True, label="Uses EF modules (MLDG only)")

        with gr.Row(equal_height=True):
            cam_method = gr.Dropdown(
                choices=list(CAM_METHODS.keys()), value="gradcam", label="CAM method",
            )
            target_class = gr.Number(
                label="Target class (optional, leave empty for predicted)",
                precision=0, value=None,
            )
            device = gr.Radio(
                choices=["cuda", "cpu"], value=default_device, label="Device",
            )

        with gr.Row(equal_height=True):
            output_dir_text = gr.Textbox(
                label="Output directory for CAMs",
                placeholder="./cam_output",
                value="./cam_output",
            )
            generate_btn = gr.Button("Generate CAMs", variant="primary")

        gallery = gr.Gallery(label="CAM Results", columns=4, height="auto")
        results_table = gr.Dataframe(label="Results Summary")

        generate_btn.click(
            fn=generate_cams,
            inputs=[
                uploaded_files, folder_dir_files,
                dataset_dir_files, dataset_split, sample_n, sample_classes, sample_seed,
                network, weights_file, num_classes, uses_ef_modules,
                cam_method, target_class, device, output_dir_text,
            ],
            outputs=[gallery, results_table],
        )

    demo.launch()


if __name__ == "__main__":
    main()
