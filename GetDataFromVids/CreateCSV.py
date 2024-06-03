# Open a file in write mode
with open('output.txt', 'w') as file:
    # Write lines from 422 to 527, each with the format "n,0"
    for n in range(422, 528):
        file.write(f"{n},3\n")
