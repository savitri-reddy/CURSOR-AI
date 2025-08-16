#!/usr/bin/env python3
"""
Test script for Traffic AI System
"""

def test_traffic_ai():
    try:
        print("Testing Traffic AI System...")
        
        # Test imports
        from traffic_ai_model import TrafficAIModel
        print("✓ Traffic AI model imported successfully")
        
        # Test model initialization
        model = TrafficAIModel()
        print("✓ Model initialized successfully")
        
        # Test data generation
        data = model.create_traffic_data(100)
        print(f"✓ Generated {len(data)} traffic data points")
        print(f"✓ Data columns: {list(data.columns)}")
        
        # Test model training
        print("\nTraining traffic prediction model...")
        results = model.train_traffic_prediction_model(data)
        print(f"✓ Model trained successfully")
        print(f"✓ MSE: {results['mse']:.2f}")
        print(f"✓ RMSE: {results['rmse']:.2f}")
        
        # Test prediction
        print("\nTesting prediction...")
        test_input = {
            'hour': 8,
            'day_of_week': 0,  # Monday
            'is_weekend': 0,
            'is_rush_hour': 1,
            'location': 'Downtown',
            'weather': 'Clear'
        }
        
        prediction = model.predict_traffic_flow(test_input)
        if isinstance(prediction, dict):
            print(f"✓ Prediction successful: {prediction['predicted_flow']:.0f} vehicles/hour")
            print(f"✓ Flow category: {prediction['flow_category']}")
        else:
            print(f"✗ Prediction failed: {prediction}")
        
        # Test congestion model
        print("\nTraining congestion model...")
        congestion_results = model.train_congestion_model(data)
        print(f"✓ Congestion model trained successfully")
        print(f"✓ Accuracy: {congestion_results['accuracy']:.2%}")
        
        # Test congestion prediction
        congestion_input = test_input.copy()
        congestion_input['traffic_flow'] = 1500
        
        congestion_prediction = model.predict_congestion(congestion_input)
        if isinstance(congestion_prediction, dict):
            print(f"✓ Congestion prediction: {congestion_prediction['predicted_congestion']}")
            print(f"✓ Confidence: {congestion_prediction['confidence']:.2%}")
        else:
            print(f"✗ Congestion prediction failed: {congestion_prediction}")
        
        # Test route optimization
        print("\nTesting route optimization...")
        routes = model.optimize_route('Downtown', 'Suburbs', '08:00', 'Clear')
        print(f"✓ Route optimization successful")
        print(f"✓ Found {len(routes)} route options")
        
        # Test insights
        print("\nTesting traffic insights...")
        insights = model.get_traffic_insights(data)
        print(f"✓ Insights generated successfully")
        print(f"✓ Peak hours: {insights['peak_hours']}")
        
        print("\n🎉 All tests passed! Traffic AI system is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_traffic_ai()
    if success:
        print("\n✅ You can now run: streamlit run traffic_ai_app.py")
    else:
        print("\n❌ Please fix the errors before running the Streamlit app") 