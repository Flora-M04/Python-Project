def city_count_dict(filename):
    city_count = {}
    with open(filename, 'r') as file:
        next(file)  # Skip the header line
        for line in file:
            parts = line.strip().split(',')  # Split on comma
            if len(parts) >= 2:
                city = parts[0].strip()
                if city in city_count:
                    city_count[city] += 1
                else:
                    city_count[city] = 1
    return city_count

filename = 'Dataset_Day1.csv'
city_counts = city_count_dict(filename)
print(city_counts)
