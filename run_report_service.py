import argparse
from report_generator import ReportGenerator

def main():
    parser = argparse.ArgumentParser(description='Run the spectrum report generator service')
    parser.add_argument('--interval', type=int, default=6, 
                        help='Interval in hours between report generation (default: 6)')
    parser.add_argument('--generate-now', action='store_true',
                        help='Generate a report immediately before starting the scheduler')
    parser.add_argument('--fine-tune', action='store_true',
                        help='Fine-tune the model before starting the service')
    
    args = parser.parse_args()
    
    # Initialize the report generator
    generator = ReportGenerator()
    
    # Fine-tune the model if requested
    if args.fine_tune:
        from report_generator import prepare_training_data
        training_data_path = prepare_training_data()
        generator.fine_tune_model(training_data_path)
    
    # Generate a report immediately if requested
    if args.generate_now:
        generator.generate_report()
    
    # Start the scheduler
    print(f"Starting report scheduler with {args.interval} hour interval...")
    generator.schedule_reports(interval_hours=args.interval)

if __name__ == "__main__":
    main()