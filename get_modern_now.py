# Save as: C:\Users\Faiz Ahmed\OneDrive\Desktop\GRID-X\get_full_modern_data.py
import fastf1 as f1
import pandas as pd
import os
import time

def get_full_modern_data():
    """Get COMPLETE modern data including 2023-2024"""
    print("🚀 GETTING FULL MODERN DATA (2021-2024)...")
    
    # Setup
    cache_path = "C:/Users/Faiz Ahmed/OneDrive/Desktop/GRID-X/data/cache/"
    output_path = "C:/Users/Faiz Ahmed/OneDrive/Desktop/GRID-X/data/raw/fastf1/"
    
    os.makedirs(cache_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    f1.Cache.enable_cache(cache_path)
    print("✅ Cache enabled")
    
    # Get ALL seasons and ALL available races
    seasons = [2021, 2022, 2023, 2024]
    
    all_laps = []
    all_sessions = []
    all_weather = []
    
    for year in seasons:
        print(f"\n📅 COLLECTING {year} SEASON...")
        
        try:
            # Try to get schedule
            schedule = f1.get_event_schedule(year)
            print(f"   Found {len(schedule)} events in {year}")
            
            for _, event in schedule.iterrows():
                try:
                    # Skip test sessions
                    if any(x in event['EventName'].lower() for x in ['test', 'pre-season']):
                        continue
                        
                    print(f"   🏎️  {event['EventName']}...")
                    
                    # Try to load session
                    session = f1.get_session(year, event['EventName'], 'R')
                    session.load(telemetry=False, weather=True, messages=False)
                    
                    # Session info
                    session_info = {
                        'year': year,
                        'round': event['RoundNumber'],
                        'event_name': event['EventName'],
                        'circuit': event['Location'],
                        'country': event['Country'],
                        'date': event['EventDate'],
                        'total_laps': len(session.laps) if hasattr(session, 'laps') and session.laps is not None else 0
                    }
                    all_sessions.append(session_info)
                    
                    # Lap data
                    if hasattr(session, 'laps') and session.laps is not None and len(session.laps) > 0:
                        laps = session.laps.copy()
                        laps['year'] = year
                        laps['round'] = event['RoundNumber']
                        laps['event_name'] = event['EventName']
                        all_laps.append(laps)
                    
                    # Weather data
                    if hasattr(session, 'weather_data') and session.weather_data is not None and len(session.weather_data) > 0:
                        weather = session.weather_data.copy()
                        weather['year'] = year
                        weather['round'] = event['RoundNumber']
                        weather['event_name'] = event['EventName']
                        all_weather.append(weather)
                    
                    print(f"      ✅ {len(session.laps) if hasattr(session, 'laps') else 0} laps")
                    time.sleep(0.3)  # Small delay to be nice to API
                    
                except Exception as e:
                    print(f"      ❌ {event['EventName']}: {str(e)[:80]}")
                    
        except Exception as e:
            print(f"   ❌ Could not get {year} schedule: {e}")
            # Try manual events for this year
            manual_events = get_manual_events(year)
            if manual_events:
                print(f"   🔧 Trying manual events for {year}...")
                for event_name in manual_events:
                    try:
                        print(f"   🏎️  {event_name}...")
                        session = f1.get_session(year, event_name, 'R')
                        session.load(telemetry=False, weather=True, messages=False)
                        
                        # [Same collection code as above...]
                        # ... (copy the session collection code from above)
                        
                    except Exception as e2:
                        print(f"      ❌ {event_name}: {str(e2)[:80]}")
    
    # SAVE EVERYTHING
    print(f"\n💾 SAVING FULL MODERN DATASET...")
    
    if all_laps:
        laps_df = pd.concat(all_laps, ignore_index=True)
        laps_path = output_path + "FULL_fastf1_lap_times_2021_2024.csv"
        laps_df.to_csv(laps_path, index=False)
        print(f"✅ Saved {len(laps_df)} laps to: {laps_path}")
    
    if all_sessions:
        sessions_df = pd.DataFrame(all_sessions)
        sessions_path = output_path + "FULL_fastf1_sessions_2021_2024.csv"
        sessions_df.to_csv(sessions_path, index=False)
        print(f"✅ Saved {len(sessions_df)} sessions to: {sessions_path}")
    
    if all_weather:
        weather_df = pd.concat(all_weather, ignore_index=True)
        weather_path = output_path + "FULL_fastf1_weather_2021_2024.csv"
        weather_df.to_csv(weather_path, index=False)
        print(f"✅ Saved {len(weather_df)} weather records to: {weather_path}")
    
    # Final summary
    print(f"\n📊 FULL MODERN DATA COLLECTION COMPLETE!")
    if all_laps:
        years_collected = laps_df['year'].unique()
        print(f"🎯 Years collected: {sorted(years_collected)}")
        print(f"🏁 Total modern laps: {len(laps_df):,}")
        print(f"📍 Files saved with 'FULL_' prefix")

def get_manual_events(year):
    """Get manual event list for years where schedule fails"""
    manual_events = {
        2023: [
            'Bahrain', 'Saudi Arabia', 'Australia', 'Azerbaijan', 'Miami',
            'Monaco', 'Spain', 'Canada', 'Austria', 'Great Britain',
            'Hungary', 'Belgium', 'Netherlands', 'Italy', 'Singapore',
            'Japan', 'Qatar', 'United States', 'Mexico City', 'Brazil',
            'Las Vegas', 'Abu Dhabi'
        ],
        2024: [
            'Bahrain', 'Saudi Arabia', 'Australia', 'Japan', 'China',
            'Miami', 'Italy', 'Monaco', 'Canada', 'Spain', 'Austria',
            'Great Britain', 'Hungary', 'Belgium', 'Netherlands', 'Italy',
            'Azerbaijan', 'Singapore', 'United States', 'Mexico City',
            'Brazil', 'Las Vegas', 'Qatar', 'Abu Dhabi'
        ]
    }
    return manual_events.get(year, [])

if __name__ == "__main__":
    get_full_modern_data()