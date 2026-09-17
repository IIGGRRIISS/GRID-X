
import fastf1 as f1
import pandas as pd
import os
import time

class FastF1DataCollectorFixed:
    def __init__(self):
        self.cache_dir = "data/cache"
        self.output_dir = "data/raw/fastf1"
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories"""
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        f1.Cache.enable_cache(self.cache_dir)
        print("✅ FastF1 cache enabled")
    
    def get_recent_seasons_fixed(self):
        """Get data for recent seasons (2021-2022) - FIXED VERSION"""
        seasons = [2021, 2022]  # Focus on available seasons
        all_data = {
            'sessions': [],
            'lap_times': [],
            'weather': []
        }
        
        for year in seasons:
            print(f"\n📅 COLLECTING {year} SEASON...")
            try:
                schedule = f1.get_event_schedule(year)
                
                for _, event in schedule.iterrows():
                    try:
                        event_data = self.collect_event_data_fixed(year, event)
                        # FIX: Proper None check
                        if event_data is not None and event_data != {}:
                            all_data['sessions'].append(event_data['session_info'])
                            all_data['lap_times'].append(event_data['lap_times'])
                            if not event_data['weather'].empty:
                                all_data['weather'].append(event_data['weather'])
                            print(f"  ✅ {event['EventName']}")
                        else:
                            print(f"  ❌ {event['EventName']}: No data collected")
                            
                        time.sleep(0.5)  # Shorter delay
                        
                    except Exception as e:
                        print(f"  ❌ {event['EventName']}: {str(e)[:50]}")
                        
            except Exception as e:
                print(f"❌ Error with {year}: {e}")
        
        return all_data
    
    def collect_event_data_fixed(self, year, event):
        """Collect data for a single event - FIXED VERSION"""
        try:
            # Skip pre-season/test sessions
            if 'test' in event['EventName'].lower() or 'pre-season' in event['EventName'].lower():
                return None
                
            # Get race session
            session = f1.get_session(year, event['EventName'], 'R')
            session.load(telemetry=False, weather=True, messages=False)  # Reduced load
            
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
            
            # Lap times - FIX: Check if laps exist
            if hasattr(session, 'laps') and session.laps is not None and len(session.laps) > 0:
                lap_times = session.laps.copy()
                lap_times['year'] = year
                lap_times['round'] = event['RoundNumber']
                lap_times['event_name'] = event['EventName']
            else:
                lap_times = pd.DataFrame()
            
            # Weather data
            weather_data = session.weather_data if hasattr(session, 'weather_data') else pd.DataFrame()
            if weather_data is not None and len(weather_data) > 0:
                weather_data = weather_data.copy()
                weather_data['year'] = year
                weather_data['round'] = event['RoundNumber']
                weather_data['event_name'] = event['EventName']
            else:
                weather_data = pd.DataFrame()
            
            return {
                'session_info': session_info,
                'lap_times': lap_times,
                'weather': weather_data
            }
            
        except Exception as e:
            print(f"    ⚠️  Could not load {event['EventName']}: {e}")
            return None
    
    def save_collected_data_fixed(self, data_dict):
        """Save all collected data - FIXED VERSION"""
        print("\n💾 SAVING COLLECTED DATA...")
        
        # Save session info
        if data_dict['sessions']:
            sessions_df = pd.DataFrame(data_dict['sessions'])
            output_path = f"{self.output_dir}/fastf1_sessions_2021_2022.csv"
            sessions_df.to_csv(output_path, index=False)
            print(f"✅ Saved {len(sessions_df)} sessions to {output_path}")
        
        # Save lap times - FIX: Filter out empty DataFrames
        if data_dict['lap_times']:
            valid_lap_dfs = [df for df in data_dict['lap_times'] if not df.empty]
            if valid_lap_dfs:
                lap_times_df = pd.concat(valid_lap_dfs, ignore_index=True)
                output_path = f"{self.output_dir}/fastf1_lap_times_2021_2022.csv"
                lap_times_df.to_csv(output_path, index=False)
                print(f"✅ Saved {len(lap_times_df)} lap times to {output_path}")
            else:
                print("❌ No valid lap time data to save")
        
        # Save weather data
        if data_dict['weather']:
            valid_weather_dfs = [df for df in data_dict['weather'] if not df.empty]
            if valid_weather_dfs:
                weather_df = pd.concat(valid_weather_dfs, ignore_index=True)
                output_path = f"{self.output_dir}/fastf1_weather_2021_2022.csv"
                weather_df.to_csv(output_path, index=False)
                print(f"✅ Saved {len(weather_df)} weather records to {output_path}")
            else:
                print("❌ No valid weather data to save")
    
    def run_collection_fixed(self):
        """Run complete data collection - FIXED VERSION"""
        print("🚀 STARTING FASTF1 DATA COLLECTION (FIXED VERSION)...")
        print("=" * 60)
        print("This will collect 2021-2022 data using existing cache")
        print("Should be FAST now (cache already built!)")
        print("=" * 60)
        
        collected_data = self.get_recent_seasons_fixed()
        self.save_collected_data_fixed(collected_data)
        
        print("\n🎉 FASTF1 DATA COLLECTION COMPLETE!")
        print("📍 Data saved to: data/raw/fastf1/")

if __name__ == "__main__":
    collector = FastF1DataCollectorFixed()
    collector.run_collection_fixed()