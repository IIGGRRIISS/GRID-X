"""
Adapters to make existing models work together seamlessly
"""

import pandas as pd
import numpy as np

class ModelAdapter:
    """Base class for model adapters"""
    
    @staticmethod
    def create_prediction_input(race_data, driver_styles):
        """Create standardized input format for all models"""
        base_features = {
            'circuit': race_data['circuit'],
            'year': race_data['year'],
            'conditions': race_data['conditions'],
            'qualifying_results': race_data['qualifying_results']
        }
        
        # Add driver style information if available
        if driver_styles is not None:
            base_features['driver_styles'] = driver_styles
        
        return base_features

class LapTimeAdapter(ModelAdapter):
    """Adapter for lap time prediction model"""
    
    @staticmethod
    def prepare_lap_time_features(race_data, driver_styles):
        """Prepare features for lap time prediction"""
        features = []
        
        for driver, grid_pos in race_data['qualifying_results'].items():
            driver_features = {
                'Driver': driver,
                'grid_position': grid_pos,
                'circuit': race_data['circuit'],
                'AirTemp': race_data['conditions']['air_temp'],
                'TrackTemp': race_data['conditions']['track_temp'],
                'Humidity': race_data['conditions']['humidity'],
                'Rainfall': race_data['conditions']['rainfall'],
                'Compound': race_data.get('tyre_compounds', {}).get(driver, 'MEDIUM')
            }
            
            # Add driver style features if available
            if driver_styles is not None and driver in driver_styles['Driver'].values:
                style_data = driver_styles[driver_styles['Driver'] == driver].iloc[0]
                driver_features.update({
                    'AggressionScore': style_data.get('AggressionScore', 0.5),
                    'ConsistencyScore': style_data.get('ConsistencyScore', 0.5),
                    'BrakingIntensity': style_data.get('BrakingIntensity', 0.5),
                    'TyrePreservation': style_data.get('TyrePreservation', 0.5),
                    'OvertakingAbility': style_data.get('OvertakingAbility', 0.5),
                    'DrivingStyle': style_data.get('style_label', 'BALANCED')
                })
            
            features.append(driver_features)
        
        return pd.DataFrame(features)

class RaceOutcomeAdapter(ModelAdapter):
    """Adapter for race outcome classification"""
    
    @staticmethod
    def prepare_race_features(race_data, driver_styles):
        """Prepare features for race outcome prediction"""
        features = []
        
        for driver, grid_pos in race_data['qualifying_results'].items():
            driver_features = {
                'Driver': driver,
                'grid_position': grid_pos,
                'year': race_data['year'],
                'circuit': race_data['circuit'],
                'AirTemp': race_data['conditions']['air_temp'],
                'TrackTemp': race_data['conditions']['track_temp'],
                'Rainfall': race_data['conditions']['rainfall'],
                'Humidity': race_data['conditions'].get('humidity', 50)
            }
            
            # Add driver style features
            if driver_styles is not None and driver in driver_styles['Driver'].values:
                style_data = driver_styles[driver_styles['Driver'] == driver].iloc[0]
                driver_features.update({
                    'driver_aggression': style_data.get('AggressionScore', 0.5),
                    'driver_consistency': style_data.get('ConsistencyScore', 0.5),
                    'driver_tyre_preservation': style_data.get('TyrePreservation', 0.5),
                    'driver_overtaking': style_data.get('OvertakingAbility', 0.5),
                    'driver_braking': style_data.get('BrakingIntensity', 0.5),
                    'driving_style': style_data.get('style_label', 'BALANCED')
                })
            
            features.append(driver_features)
        
        return pd.DataFrame(features)

class DriverStyleAdapter(ModelAdapter):
    """Adapter for driver style analysis"""
    
    @staticmethod
    def extract_style_insights(driver_profiles, drivers=None):
        """Extract actionable insights from driver style data"""
        if driver_profiles is None:
            return {}
        
        if drivers is not None:
            driver_profiles = driver_profiles[driver_profiles['Driver'].isin(drivers)]
        
        insights = {}
        
        for _, driver_data in driver_profiles.iterrows():
            driver = driver_data['Driver']
            style = driver_data.get('style_label', 'UNKNOWN')
            aggression = driver_data.get('AggressionScore', 0.5)
            consistency = driver_data.get('ConsistencyScore', 0.5)
            tyre_preservation = driver_data.get('TyrePreservation', 0.5)
            overtaking = driver_data.get('OvertakingAbility', 0.5)
            braking = driver_data.get('BrakingIntensity', 0.5)
            
            # Generate insights based on style characteristics
            style_insights = []
            
            if aggression > 0.7:
                style_insights.append("Tends to be aggressive with overtaking and braking")
                if braking > 0.7:
                    style_insights.append("Late braking specialist")
            elif aggression < 0.3:
                style_insights.append("Conservative approach to racing")
                
            if consistency > 0.8:
                style_insights.append("Highly consistent lap times")
            elif consistency < 0.4:
                style_insights.append("Variable performance across laps")
            
            if tyre_preservation > 0.8:
                style_insights.append("Excellent tyre management")
            elif tyre_preservation < 0.4:
                style_insights.append("May struggle with tyre wear in long stints")
            
            if overtaking > 0.8:
                style_insights.append("Strong overtaking ability")
            
            if style == 'AGGRESSIVE':
                style_insights.append("May struggle with tyre preservation in long stints")
                style_insights.append("Good in wet conditions and safety car restarts")
            elif style == 'SMOOTH':
                style_insights.append("Excellent at managing tyres and fuel")
                style_insights.append("Consistent in dry conditions")
            elif style == 'OPPORTUNISTIC':
                style_insights.append("Strong racecraft and overtaking ability")
                style_insights.append("Adaptable to changing conditions")
            
            insights[driver] = {
                'style': style,
                'aggression_level': aggression,
                'consistency_level': consistency,
                'tyre_preservation': tyre_preservation,
                'overtaking_ability': overtaking,
                'braking_intensity': braking,
                'insights': style_insights
            }
        
        return insights

    @staticmethod
    def get_strategy_recommendations(driver_insights, track_type, conditions):
        """Get strategy recommendations based on driver styles and conditions"""
        recommendations = {}
        
        for driver, insight in driver_insights.items():
            driver_recs = []
            style = insight['style']
            aggression = insight['aggression_level']
            consistency = insight['consistency_level']
            tyre_pres = insight['tyre_preservation']
            
            # Track-specific recommendations
            if track_type in ['Monaco', 'Hungaroring', 'Singapore']:
                if aggression > 0.7:
                    driver_recs.append("Good for street circuits with aggressive driving")
                if consistency > 0.8:
                    driver_recs.append("Technical circuits suit consistent driving style")
            
            elif track_type in ['Monza', 'Spa', 'Baku']:
                if aggression > 0.7:
                    driver_recs.append("High-speed circuits favor aggressive overtaking")
            
            # Conditions-based recommendations
            if conditions.get('rainfall', 0) > 0:
                if aggression > 0.6:
                    driver_recs.append("Wet conditions favor aggressive drivers")
                else:
                    driver_recs.append("Be cautious in wet conditions")
            else:
                if consistency > 0.8 and tyre_pres > 0.7:
                    driver_recs.append("Ideal for one-stop strategy")
                elif aggression > 0.7:
                    driver_recs.append("Consider two-stop aggressive strategy")
            
            # Style-specific recommendations
            if style == 'SMOOTH':
                driver_recs.append("Focus on tyre preservation and consistent pace")
                driver_recs.append("Good candidate for longer stints")
            elif style == 'AGGRESSIVE':
                driver_recs.append("Push early, undercut opportunities")
                driver_recs.append("Monitor tyre wear carefully")
            elif style == 'OPPORTUNISTIC':
                driver_recs.append("Flexible strategy, adapt to race situations")
                driver_recs.append("Good at capitalizing on safety cars")
            
            recommendations[driver] = driver_recs
        
        return recommendations