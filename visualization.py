import folium

def create_map(center, zoom_start=15):
    return folium.Map(location=center, zoom_start=zoom_start)

def add_marker(map_obj, location, popup_text):
    folium.Marker(location=location, popup=popup_text).add_to(map_obj)

def save_map(map_obj, file_path):
    map_obj.save(file_path)