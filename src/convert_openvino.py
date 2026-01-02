import openvino as ov

def main():
    # Convert ONNX → OpenVINO model
    ov_model = ov.convert_model("peta_attributes.onnx")

    # Save IR files (.xml + .bin)
    ov.save_model(ov_model, "peta_attributes_openvino.xml")

    print("OpenVINO IR successfully exported:")
    print(" - peta_attributes_openvino.xml")
    print(" - peta_attributes_openvino.bin")

if __name__ == "__main__":
    main()
