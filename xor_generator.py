import csv
import random

def generate_xor_data(height):
    data = []
    for i in range(height):
        input1 = i % 2
        input2 = (i // 2) % 2
        output = int(input1 != input2)
        input1 -= random.random()*0.8-0.5
        input2 -= random.random()*0.8-0.5
        data.append([input1, input2, '', output])
    return data

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
