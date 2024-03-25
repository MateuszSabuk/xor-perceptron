import csv
import random

def generate_xor_data(height):
    data = []
    for i in range(height):
        input1 = i % 2
        input2 = (i // 2) % 2
        output = int(input1 != input2)
        data.append([randomize_input(input1), randomize_input(input2), '', int(not output), output])
    return data

def randomize_input(input_val):
    return input_val + random.random() - 0.5

def write_to_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

def main():
    height = int(input("Enter the height of the CSV file: "))
    filename = 'inputs_outputs.csv'
    data = generate_xor_data(height)
    write_to_csv(data, filename)
    print(f"CSV file '{filename}' with {height} rows generated successfully.")

if __name__ == "__main__":
    main()
