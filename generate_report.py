from report_generator import ReportGenerator

def main():
    generator = ReportGenerator()
    report, report_path = generator.generate_report()
    
    print(f"\nReport generated and saved to {report_path}")
    print("\nReport content:")
    print("-" * 80)
    print(report)
    print("-" * 80)

if __name__ == "__main__":
    main()