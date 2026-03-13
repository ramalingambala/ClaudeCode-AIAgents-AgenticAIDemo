import random


def get_weather(city: str) -> dict:
    """Get the current weather for a given city.

    Args:
        city: The name of the city to get weather for.

    Returns:
        A dictionary with weather information.
    """
    # Simulated weather data for demo purposes
    conditions = ["Sunny", "Partly Cloudy", "Cloudy", "Rainy", "Windy"]
    temp = random.randint(55, 95)
    humidity = random.randint(30, 90)

    return {
        "city": city,
        "temperature_f": temp,
        "temperature_c": round((temp - 32) * 5 / 9, 1),
        "condition": random.choice(conditions),
        "humidity_percent": humidity,
    }
