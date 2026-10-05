import os
import shutil
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

# YOUR ORIGINAL FALL DATASET
OLD_DATASET = Path(r"D:\accident\accident\fall_dataset")

# NEW CAUCAFall YOLOv8 DATASET
CAUCA_DATASET = Path(r"D:\CAUCAFall_v2.v1i.yolov8")

# NEW COMBINED DATASET
OUTPUT_DATASET = Path(r"D:\Fall_Expanded")


# ============================================================
# SETTINGS
# ============================================================

SPLITS = ["train", "valid", "test"]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_images(folder):
    """Return all image files inside a folder."""
    if not folder.exists():
        return []

    return [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def copy_old_dataset():
    """
    Copy the original dataset exactly as it is.
    This creates the base of the expanded dataset.
    """

    print("\n========================================")
    print("COPYING ORIGINAL FALL DATASET")
    print("========================================")

    for split in SPLITS:

        old_images = OLD_DATASET / split / "images"
        old_labels = OLD_DATASET / split / "labels"

        new_images = OUTPUT_DATASET / split / "images"
        new_labels = OUTPUT_DATASET / split / "labels"

        new_images.mkdir(parents=True, exist_ok=True)
        new_labels.mkdir(parents=True, exist_ok=True)

        images = find_images(old_images)

        print(f"\n{split}: {len(images)} original images")

        for image in images:

            destination_image = new_images / image.name

            # Original filename gets an OLD_ prefix
            destination_image = new_images / f"OLD_{image.name}"

            shutil.copy2(image, destination_image)

            label = old_labels / f"{image.stem}.txt"

            destination_label = new_labels / f"OLD_{image.stem}.txt"

            if label.exists():
                shutil.copy2(label, destination_label)
            else:
                # Create empty label if none exists
                destination_label.write_text("")


# ============================================================
# PROCESS CAUCAFall
# ============================================================

def process_caucafall():
    """
    Add CAUCAFall to the combined dataset.

    CAUCAFall classes:
        0 = fall
        1 = nofall

    Our final dataset:
        0 = Fall-Detected

    Therefore:
        fall   -> class 0
        nofall -> empty label
    """

    print("\n========================================")
    print("ADDING CAUCAFall")
    print("========================================")

    total_images = 0
    fall_images = 0
    nofall_images = 0
    missing_labels = 0

    for split in SPLITS:

        caucafall_images = CAUCA_DATASET / split / "images"
        caucafall_labels = CAUCA_DATASET / split / "labels"

        output_images = OUTPUT_DATASET / split / "images"
        output_labels = OUTPUT_DATASET / split / "labels"

        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)

        images = find_images(caucafall_images)

        print(f"\n{split}: {len(images)} CAUCAFall images")

        for image in images:

            total_images += 1

            # Prefix prevents filename collisions
            new_image_name = f"CAUCA_{image.name}"

            destination_image = output_images / new_image_name

            shutil.copy2(image, destination_image)

            source_label = caucafall_labels / f"{image.stem}.txt"

            destination_label = (
                output_labels / f"CAUCA_{image.stem}.txt"
            )

            # ------------------------------------------------
            # No annotation file
            # ------------------------------------------------

            if not source_label.exists():

                destination_label.write_text("")

                missing_labels += 1
                continue

            # ------------------------------------------------
            # Read CAUCAFall YOLO annotations
            # ------------------------------------------------

            lines = source_label.read_text().splitlines()

            new_lines = []

            for line in lines:

                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                if len(parts) != 5:
                    print(
                        f"WARNING: Invalid label format: "
                        f"{source_label}"
                    )
                    continue

                try:
                    class_id = int(parts[0])
                except ValueError:
                    continue

                # --------------------------------------------
                # CAUCAFall class 0 = FALL
                # Keep it as our class 0
                # --------------------------------------------

                if class_id == 0:

                    new_lines.append(
                        "0 " + " ".join(parts[1:])
                    )

                    fall_images += 1

                # --------------------------------------------
                # CAUCAFall class 1 = NOFALL
                #
                # Do NOT create a "nofall" object.
                # Empty label means negative image.
                # --------------------------------------------

                elif class_id == 1:

                    nofall_images += 1

                else:

                    print(
                        f"WARNING: Unknown class {class_id} "
                        f"in {source_label}"
                    )

            # Write converted label
            destination_label.write_text(
                "\n".join(new_lines)
            )

    return (
        total_images,
        fall_images,
        nofall_images,
        missing_labels
    )


# ============================================================
# CREATE DATA.YAML
# ============================================================

def create_data_yaml():

    yaml_content = """path: D:/Fall_Expanded
train: train/images
val: valid/images
test: test/images

nc: 1
names:
  0: Fall-Detected
"""

    yaml_file = OUTPUT_DATASET / "data.yaml"

    yaml_file.write_text(yaml_content)

    print("\nCreated:")
    print(yaml_file)


# ============================================================
# DATASET VALIDATION
# ============================================================

def validate_dataset():

    print("\n========================================")
    print("VALIDATING DATASET")
    print("========================================")

    total_images = 0
    total_labels = 0
    total_objects = 0
    empty_labels = 0

    for split in SPLITS:

        images_folder = OUTPUT_DATASET / split / "images"
        labels_folder = OUTPUT_DATASET / split / "labels"

        images = find_images(images_folder)

        split_objects = 0
        split_empty = 0

        for image in images:

            total_images += 1

            label = labels_folder / f"{image.stem}.txt"

            if not label.exists():
                print(
                    f"WARNING: Missing label: {label}"
                )
                continue

            total_labels += 1

            text = label.read_text().strip()

            if not text:
                empty_labels += 1
                split_empty += 1
                continue

            for line in text.splitlines():

                parts = line.split()

                if len(parts) == 5:

                    try:
                        class_id = int(parts[0])

                        if class_id != 0:
                            print(
                                f"WARNING: Invalid class "
                                f"{class_id}: {label}"
                            )

                        total_objects += 1
                        split_objects += 1

                    except ValueError:
                        print(
                            f"WARNING: Invalid class ID: {label}"
                        )

        print(
            f"{split}: "
            f"{len(images)} images | "
            f"{split_objects} fall objects | "
            f"{split_empty} negative images"
        )

    print("\n----------------------------------------")
    print("TOTAL")
    print("----------------------------------------")
    print(f"Images       : {total_images}")
    print(f"Label files  : {total_labels}")
    print(f"Fall objects : {total_objects}")
    print(f"Empty labels : {empty_labels}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("========================================")
    print(" FALL DATASET MERGER")
    print("========================================")

    print(f"\nOriginal dataset:")
    print(OLD_DATASET)

    print(f"\nCAUCAFall dataset:")
    print(CAUCA_DATASET)

    print(f"\nOutput dataset:")
    print(OUTPUT_DATASET)

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if not OLD_DATASET.exists():

        print("\nERROR:")
        print("Original fall dataset was not found.")
        print(OLD_DATASET)
        return

    if not CAUCA_DATASET.exists():

        print("\nERROR:")
        print("CAUCAFall dataset was not found.")
        print(CAUCA_DATASET)
        return

    # --------------------------------------------------------
    # Prevent accidental overwrite
    # --------------------------------------------------------

    if OUTPUT_DATASET.exists():

        print("\nERROR:")
        print("Fall_Expanded already exists.")

        print(
            "\nDelete/rename that folder first if you want "
            "to create a fresh merge."
        )

        return

    # --------------------------------------------------------
    # Create output
    # --------------------------------------------------------

    OUTPUT_DATASET.mkdir(parents=True)

    # --------------------------------------------------------
    # Copy original dataset
    # --------------------------------------------------------

    copy_old_dataset()

    # --------------------------------------------------------
    # Add CAUCAFall
    # --------------------------------------------------------

    results = process_caucafall()

    (
        total_images,
        fall_images,
        nofall_images,
        missing_labels
    ) = results

    # --------------------------------------------------------
    # Create YAML
    # --------------------------------------------------------

    create_data_yaml()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_dataset()

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n========================================")
    print("MERGE COMPLETE")
    print("========================================")

    print(f"\nCAUCAFall images added : {total_images}")
    print(f"Fall annotations       : {fall_images}")
    print(f"No-fall images         : {nofall_images}")
    print(f"Missing labels         : {missing_labels}")

    print("\nFinal dataset:")
    print(OUTPUT_DATASET)

    print("\nData YAML:")
    print(OUTPUT_DATASET / "data.yaml")

    print("\nIMPORTANT:")
    print("Your original dataset has NOT been modified.")
    print("Your original model has NOT been modified.")


if __name__ == "__main__":
    main()