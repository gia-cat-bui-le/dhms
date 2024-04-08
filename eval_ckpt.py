import torch

def view_and_save_pt_content(file_path, output_file):
    try:
        content = torch.load(file_path, map_location='cpu')
        with open(output_file, 'w') as f:
            f.write(str(content))
        print(f"Content of '{file_path}' saved to '{output_file}' successfully.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Example usage
file_path = 'save/pcmdm_cont/opt000100000.pt'  # Change this to your .pt file path
output_file = 'content.txt'  # Output text file name
view_and_save_pt_content(file_path, output_file)
