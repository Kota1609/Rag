def filter_urls(input_file, output_file, unwanted_suffix):
    # Read URLs from the input file
    with open(input_file, "r") as file:
        urls = file.read().splitlines()

    # Filter URLs that do not end with the unwanted suffix
    filtered_urls = [url for url in urls if not url.endswith(unwanted_suffix)]

    # Write the filtered URLs to the output file
    with open(output_file, "w") as file:
        for url in filtered_urls:
            file.write(url + "\n")

    print(f"Filtered URLs have been written to {output_file}")

# Specify the input and output file paths
input_file = "cs_found_links.txt"
output_file = "cs_filtered_links.txt"
unwanted_suffix = "#main-content"

# Filter the URLs
filter_urls(input_file, output_file, unwanted_suffix)
