import streamlit as st
import leafmap

def get_stac_data(query):
    """Simulate STAC API call with hardcoded responses"""
    if "snow cover" in query.lower() and "himalaya" in query.lower():
        return {
            "type": "FeatureCollection",
            "features": [{
                "assets": {
                    "visual": {"href": "https://raw.githubusercontent.com/opengeos/datasets/main/samples/snow_cover_himalayas.tif"}
                }
            }]
        }
    elif "chlorophyll" in query.lower():
        return {
            "type": "FeatureCollection",
            "features": [{
                "assets": {
                    "visual": {"href": "https://raw.githubusercontent.com/opengeos/datasets/main/samples/chl_concentration.tif"}
                }
            }]
        }
    return None

def render_simple_map():
    """Fallback map without geopandas"""
    m = leafmap.Map(center=[20, 77], zoom=5)
    m.add_basemap("SATELLITE")
    return m