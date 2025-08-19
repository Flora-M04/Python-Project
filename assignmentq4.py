def apollo_cities_set(filename):
    apollo_cities = set()
    try:
        with open(filename, 'r') as file:
            next(file)  # Skip header
            for line in file:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    city = parts[0].strip()
                    hospital = parts[1].strip().lower()
                    if "apollo" in hospital:
                        apollo_cities.add(city)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    return apollo_cities
if __name__ == "__main__":
    filename = 'Dataset_Day1.csv'
    apollo_cities = apollo_cities_set(filename)
    print(apollo_cities)
    print("Problem 4: Cities with at least one Apollo hospital")
    for city in apollo_cities:
        print(city)