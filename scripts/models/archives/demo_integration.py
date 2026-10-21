"""
Demo script showing how to use the integrated predictor
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path("C:/Users/Faiz Ahmed/OneDrive/Desktop/GRID-X")
sys.path.append(str(project_root))

try:
    from scripts.models.integrated_predictor import GridXIntegratedPredictor
    from scripts.models.model_adapters import DriverStyleAdapter
    print("✅ Successfully imported required modules")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("🔄 Trying direct import...")
    
    # Try direct import
    import importlib.util
    
    # Import integrated predictor
    spec_pred = importlib.util.spec_from_file_location(
        "integrated_predictor",
        project_root / "scripts" / "models" / "integrated_predictor.py"
    )
    integrated_predictor = importlib.util.module_from_spec(spec_pred)
    spec_pred.loader.exec_module(integrated_predictor)
    GridXIntegratedPredictor = integrated_predictor.GridXIntegratedPredictor
    
    # Import model adapters
    spec_adapt = importlib.util.spec_from_file_location(
        "model_adapters",
        project_root / "scripts" / "models" / "model_adapters.py"
    )
    model_adapters = importlib.util.module_from_spec(spec_adapt)
    spec_adapt.loader.exec_module(model_adapters)
    DriverStyleAdapter = model_adapters.DriverStyleAdapter

def demo_basic_usage():
    """Demonstrate basic usage of the integrated system"""
    
    print("🚀 GRID-X INTEGRATED PREDICTOR DEMO")
    print("=" * 50)
    
    # Initialize predictor
    predictor = GridXIntegratedPredictor()
    
    # Define your race
    my_race = {
        'circuit': 'Silverstone',
        'year': 2024,
        'qualifying_results': {
            'VER': 1, 'HAM': 2, 'LEC': 3, 'NOR': 4, 
            'ALO': 5, 'RUS': 6, 'SAI': 7, 'PIA': 8
        },
        'conditions': {
            'air_temp': 18,
            'track_temp': 25,
            'humidity': 70,
            'rainfall': 0.1  # Light rain
        },
        'tyre_compounds': {
            'VER': 'INTERMEDIATE', 'HAM': 'INTERMEDIATE', 'LEC': 'INTERMEDIATE',
            'NOR': 'INTERMEDIATE', 'ALO': 'INTERMEDIATE', 'RUS': 'INTERMEDIATE',
            'SAI': 'INTERMEDIATE', 'PIA': 'INTERMEDIATE'
        }
    }
    
    print("📋 RACE SCENARIO:")
    print(f"   Circuit: {my_race['circuit']}")
    print(f"   Conditions: {my_race['conditions']}")
    print(f"   Qualifying: {my_race['qualifying_results']}")
    
    # Get integrated prediction
    results = predictor.integrated_race_prediction(my_race)
    
    return results

def demo_dry_race():
    """Demo for dry race conditions"""
    
    predictor = GridXIntegratedPredictor()
    
    dry_race = {
        'circuit': 'Monza',
        'year': 2024,
        'qualifying_results': {
            'VER': 1, 'LEC': 2, 'SAI': 3, 'NOR': 4,
            'HAM': 5, 'RUS': 6, 'ALO': 7, 'PIA': 8
        },
        'conditions': {
            'air_temp': 30,
            'track_temp': 45,
            'humidity': 50,
            'rainfall': 0  # Dry conditions
        },
        'tyre_compounds': {
            'VER': 'SOFT', 'LEC': 'SOFT', 'SAI': 'MEDIUM', 'NOR': 'SOFT',
            'HAM': 'MEDIUM', 'RUS': 'MEDIUM', 'ALO': 'HARD', 'PIA': 'SOFT'
        }
    }
    
    print("\n🏎️ DRY RACE SCENARIO - Monza")
    print("=" * 40)
    
    results = predictor.integrated_race_prediction(dry_race)
    return results

def demo_driver_style_insights():
    """Demo driver style insights and strategy recommendations"""
    
    predictor = GridXIntegratedPredictor()
    
    # Get driver styles
    drivers = ['HAM', 'VER', 'ALO', 'LEC', 'NOR', 'RUS']
    driver_styles = predictor.get_driver_styles(drivers)
    
    print("\n🎯 DRIVER STYLE INSIGHTS")
    print("=" * 40)
    
    # Extract insights
    insights = DriverStyleAdapter.extract_style_insights(driver_styles)
    
    for driver, insight in insights.items():
        print(f"\n{driver}:")
        print(f"  Style: {insight['style']}")
        print(f"  Aggression: {insight['aggression_level']:.2f}")
        print(f"  Consistency: {insight['consistency_level']:.2f}")
        print(f"  Tyre Preservation: {insight['tyre_preservation']:.2f}")
        print(f"  Overtaking: {insight['overtaking_ability']:.2f}")
        print("  Insights:")
        for insight_text in insight['insights']:
            print(f"    • {insight_text}")
    
    # Get strategy recommendations
    recommendations = DriverStyleAdapter.get_strategy_recommendations(
        insights, 
        track_type='Monza', 
        conditions={'rainfall': 0}
    )
    
    print(f"\n🏁 STRATEGY RECOMMENDATIONS FOR MONZA:")
    print("=" * 40)
    for driver, recs in recommendations.items():
        print(f"\n{driver}:")
        for rec in recs:
            print(f"  • {rec}")

def display_results(results):
    """Display prediction results in a formatted way"""
    
    print("\n" + "="*60)
    print("🎯 PREDICTION RESULTS")
    print("="*60)
    
    print(f"\n🏁 RACE: {results['race_info']['circuit']} {results['race_info']['year']}")
    print(f"📊 Conditions: {results['race_info']['conditions']}")
    
    print(f"\n🏆 FAVORITE: {results['summary']['favorite']}")
    if 'predicted_best_lap' in results['summary']:
        print(f"⏱️ PREDICTED BEST LAP: {results['summary']['predicted_best_lap']}")
    
    print(f"\n📈 DETAILED PREDICTIONS:")
    print("-" * 50)
    
    # Sort drivers by win probability
    sorted_drivers = sorted(
        results['race_outcome_predictions'].items(),
        key=lambda x: x[1]['win_probability'],
        reverse=True
    )
    
    for driver, prediction in sorted_drivers:
        style_info = results['driver_analysis'].get(driver, {})
        style = style_info.get('driving_style', 'UNKNOWN')
        
        # Get lap time if available
        lap_time = "N/A"
        if driver in results['lap_time_predictions']:
            lap_time = f"{results['lap_time_predictions'][driver]['predicted_lap_time']}s"
        
        print(f"\n   {driver}:")
        print(f"      📍 Grid: P{prediction['grid_position']}")
        print(f"      🎯 Style: {style}")
        print(f"      ⏱️ Predicted Lap: {lap_time}")
        print(f"      🏆 Win: {prediction['win_probability']*100:.1f}%")
        print(f"      🥇 Podium: {prediction['podium_probability']*100:.1f}%")
        print(f"      📊 Points: {prediction['points_finish_probability']*100:.1f}%")
        
        # Show style characteristics
        if style != 'UNKNOWN':
            print(f"      🔥 Aggression: {style_info.get('aggression', 0.5):.2f}")
            print(f"      📈 Consistency: {style_info.get('consistency', 0.5):.2f}")
            print(f"      🛞 Tyre Preservation: {style_info.get('tyre_preservation', 0.5):.2f}")

if __name__ == "__main__":
    # Run all demos
    try:
        print("🚀 GRID-X INTEGRATED SYSTEM DEMONSTRATION")
        print("=" * 60)
        
        # Demo 1: Basic usage with wet conditions
        results1 = demo_basic_usage()
        display_results(results1)
        
        # Demo 2: Dry race scenario
        results2 = demo_dry_race()
        display_results(results2)
        
        # Demo 3: Driver style insights
        demo_driver_style_insights()
        
        print("\n" + "="*60)
        print("✅ INTEGRATION DEMO COMPLETE!")
        print("🎯 You now have a working 3-model integrated system!")
        print("🚀 System successfully combines:")
        print("   • Driver Style Analysis")
        print("   • Lap Time Prediction") 
        print("   • Race Outcome Prediction")
        print("   • Strategy Recommendations")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        print("💡 Make sure all files are in the correct location:")
        print("   - scripts/models/integrated_predictor.py")
        print("   - scripts/models/model_adapters.py") 
        print("   - scripts/models/demo_integration.py")