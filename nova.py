#!/usr/bin/env python3
"""
Unified Nova HR Assistant - CV Analysis Tool

This script can process single CV files or entire directories of CVs,
providing comprehensive analysis with intelligent candidate evaluation.
"""

import sys
import json
import logging
import time
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.nova_graph import NovaGraph, NovaGraphError
from src.config.config_loader import ConfigLoader
from src.utils.file_parser import FileParser, FileParsingError
from src.utils.rate_limiter import get_rate_limiter


def setup_logging(debug_mode: bool = False, batch_mode: bool = False) -> None:
    """Set up optimized logging for single or batch processing"""
    log_level = logging.DEBUG if debug_mode else logging.INFO
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # File handler - different log files for single vs batch
    log_file = 'nova_batch.log' if batch_mode else 'nova.log'
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Reduce noise from external libraries
    for lib in ['httpx', 'openai', 'anthropic', 'google', 'groq', 'urllib3']:
        logging.getLogger(lib).setLevel(logging.WARNING)


def print_progress_header(batch_mode: bool = False):
    """Print analysis progress header"""
    print("\n" + "=" * 70)
    if batch_mode:
        print("🚀 NOVA HR ASSISTANT - BATCH PROCESSING")
    else:
        print("🚀 NOVA HR ASSISTANT - CV ANALYSIS")
    print("=" * 70)


def print_rate_limit_info():
    """Print current rate limiting information"""
    rate_limiter = get_rate_limiter()
    stats = rate_limiter.get_stats()
    
    print(f"📊 Rate Limiting: {stats['requests_in_last_minute']}/{stats['max_requests_per_minute']} requests/min")
    print(f"⏱️  Request delay: {stats['request_delay']}s between calls")


def print_analysis_progress(step: str, total_steps: int = 7):
    """Print analysis step progress"""
    steps = [
        "CV Parser",
        "Experience Analyzer", 
        "Skills Analyzer",
        "Education Analyzer",
        "Certification Analyzer",
        "Suitability Scorer",
        "Report Compiler"
    ]
    
    try:
        current_step = steps.index(step) + 1
        progress = (current_step / total_steps) * 100
        
        print(f"\n🔄 Step {current_step}/{total_steps}: {step}")
        print(f"📈 Progress: {progress:.1f}%")
        
        # Show progress bar
        bar_length = 30
        filled_length = int(bar_length * current_step // total_steps)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f"[{bar}] {current_step}/{total_steps}")
        
    except ValueError:
        print(f"🔄 Processing: {step}")


def print_final_summary(result: dict, processing_time: float):
    """Print final analysis summary"""
    print("\n" + "=" * 70)
    print("📊 ANALYSIS COMPLETE")
    print("=" * 70)
    
    if result.get('error'):
        print(f"❌ Status: FAILED")
        print(f"🚨 Error: {result.get('error_message', 'Unknown error')}")
        return
    
    # Basic info
    candidate_info = result.get('candidate', {})
    candidate_name = candidate_info.get('name', 'Unknown')
    suitability_score = result.get('evaluation', {}).get('suitability_score', 0)
    
    print(f"✅ Status: SUCCESS")
    print(f"👤 Candidate: {candidate_name}")
    print(f"📈 Suitability Score: {suitability_score}/100")
    print(f"⏱️  Processing Time: {processing_time:.1f} seconds")
    
    # Print detailed score breakdown
    evaluation = result.get('evaluation', {})
    scoring_details = evaluation.get('scoring_details', {})
    if scoring_details:
        component_scores = scoring_details.get('component_scores', {})
        scoring_weights = scoring_details.get('scoring_weights', {})
        potential_score = scoring_details.get('potential_score', suitability_score)
        
        print(f"\n📊 DETAILED SCORE BREAKDOWN:")
        print(f"   Current Score: {suitability_score}/100")
        print(f"   Potential Score: {potential_score}/100")
        
        if component_scores and scoring_weights:
            print(f"   Component Scores:")
            for component, score in component_scores.items():
                weight = scoring_weights.get(component, 0.0)
                weighted_contribution = score * weight
                print(f"      • {component.title()}: {score}/100 (weight: {weight:.1%}, contribution: {weighted_contribution:.1f})")
        else:
            print(f"   Component Scores:")
            for component, score in component_scores.items():
                print(f"      • {component.title()}: {score}/100")
    
    # Rate limiting stats
    rate_limiter = get_rate_limiter()
    final_stats = rate_limiter.get_stats()
    print(f"📊 API Calls Made: {final_stats['requests_in_last_minute']}")
    
    # Key insights
    evaluation = result.get('evaluation', {})
    strengths = evaluation.get('strengths', [])
    if strengths:
        print(f"\n💪 Top Strengths:")
        for i, strength in enumerate(strengths[:3], 1):
            print(f"   {i}. {strength}")
    
    gaps = evaluation.get('areas_for_improvement', [])
    if gaps:
        print(f"\n⚠️  Areas for Improvement:")
        for i, gap in enumerate(gaps[:3], 1):
            print(f"   {i}. {gap}")


def sanitize_filename(name: str) -> str:
    """
    Convert candidate name to a safe filename format.
    
    Args:
        name: Candidate's full name
        
    Returns:
        Sanitized filename with lowercase and underscores
    """
    if not name or not name.strip():
        return "unknown_candidate"
    
    # Convert to lowercase and replace spaces/special chars with underscores
    sanitized = re.sub(r'[^\w\s-]', '', name.lower())
    sanitized = re.sub(r'[-\s]+', '_', sanitized)
    
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    
    # Ensure it's not empty
    if not sanitized:
        return "unknown_candidate"
    
    return sanitized


def get_candidate_name_from_cv(cv_text: str, nova_graph: NovaGraph) -> str:
    """
    Extract candidate name from CV text using Nova's CV parser.
    
    Args:
        cv_text: Extracted CV text content
        nova_graph: Nova graph instance for parsing
        
    Returns:
        Candidate name or "unknown_candidate" if extraction fails
    """
    try:
        # Use Nova's CV parser to extract candidate info
        from src.models.state import CVAnalysisState
        from src.utils.llm_manager import LLMManager
        
        # Create a minimal state for name extraction
        state = CVAnalysisState(cv_text=cv_text, position="")
        
        # Get LLM manager from config
        config = ConfigLoader.load_nova_config()
        llm_manager = LLMManager(config.llm)
        
        # Import and use CV parser node
        from src.nodes.cv_parser import cv_parser_node
        
        # Extract candidate info
        result = cv_parser_node(state, llm_manager)
        
        if result.get('candidate_info') and result['candidate_info'].name:
            return result['candidate_info'].name.strip()
        
    except Exception as e:
        logging.warning(f"Failed to extract candidate name: {str(e)}")
    
    return "unknown_candidate"


def find_cv_files(directory: str) -> List[Path]:
    """
    Find all supported CV files in the directory.
    
    Args:
        directory: Path to directory containing CV files
        
    Returns:
        List of Path objects for supported CV files
    """
    directory_path = Path(directory)
    
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    if not directory_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory}")
    
    cv_files = []
    supported_extensions = FileParser.SUPPORTED_EXTENSIONS
    
    for file_path in directory_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            cv_files.append(file_path)
    
    return sorted(cv_files)


def create_results_directory() -> Path:
    """
    Create the results directory if it doesn't exist.
    
    Returns:
        Path to the results directory
    """
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    return results_dir


def process_single_cv(cv_file: Path, position_description: str, nova_graph: NovaGraph, 
                     results_dir: Optional[Path] = None, delay_seconds: int = 0) -> Dict[str, Any]:
    """
    Process a single CV file and save results.
    
    Args:
        cv_file: Path to the CV file
        position_description: Job position description
        nova_graph: Nova graph instance
        results_dir: Directory to save results (None for single file mode)
        delay_seconds: Delay after processing (for batch mode)
        
    Returns:
        Dictionary with processing results
    """
    result_info = {
        "file": str(cv_file),
        "success": False,
        "candidate_name": "unknown",
        "output_file": None,
        "error": None,
        "processing_time": 0
    }
    
    start_time = time.time()
    
    try:
        print(f"\n📖 Processing: {cv_file.name}")
        
        # Extract text from CV
        cv_text = FileParser.extract_text(str(cv_file))
        
        if len(cv_text.strip()) < 50:
            raise ValueError("CV text too short (minimum 50 characters)")
        
        # Extract candidate name
        candidate_name = get_candidate_name_from_cv(cv_text, nova_graph)
        result_info["candidate_name"] = candidate_name
        
        print(f"👤 Candidate: {candidate_name}")
        
        # Process CV through Nova
        analysis_result = nova_graph.process_candidate(cv_text, position_description)
        
        # Determine output file location and name
        if results_dir:
            # Batch mode - save in results directory with candidate name
            filename = sanitize_filename(candidate_name)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = results_dir / f"{filename}_{timestamp}.json"
        else:
            # Single file mode - save in current directory with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = Path(f"nova_analysis_{timestamp}.json")
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, indent=2, ensure_ascii=False)
        
        result_info["success"] = True
        result_info["output_file"] = str(output_file)
        
        # Print enhanced summary
        evaluation = analysis_result.get('evaluation', {}) or {}
        suitability_score = evaluation.get('suitability_score', 0)
        scoring_details = evaluation.get('scoring_details', {}) or {}
        
        # Enhanced completion summary
        print(f"Completed - Score: {suitability_score}/100")
        
        # Show enhanced metrics if available
        if scoring_details:
            potential_score = scoring_details.get('potential_score', suitability_score)
            confidence_level = scoring_details.get('confidence_level', 0.5)
            risk_factors = scoring_details.get('risk_factors', [])
            development_timeline = scoring_details.get('development_timeline', '')
            component_scores = scoring_details.get('component_scores', {})
            scoring_weights = scoring_details.get('scoring_weights', {})
            
            print(f"Potential Score: {potential_score}/100")
            print(f"Confidence: {confidence_level:.1%}")
            print(f"Risk Factors: {len(risk_factors)} identified")
            if development_timeline:
                print(f"Development: {development_timeline}")
            
            # Print component score breakdown
            if component_scores:
                print(f"Component Scores:")
                if scoring_weights:
                    for component, score in component_scores.items():
                        weight = scoring_weights.get(component, 0.0)
                        weighted_contribution = score * weight
                        print(f"  • {component.title()}: {score}/100 (weight: {weight:.1%}, contribution: {weighted_contribution:.1f})")
                else:
                    for component, score in component_scores.items():
                        print(f"  • {component.title()}: {score}/100")
            
            # Show bias detection if any
            bias_detection = scoring_details.get('bias_detection', {})
            if bias_detection.get('bias_detected', False):
                total_biases = bias_detection.get('total_biases', 0)
                print(f"Bias Detection: {total_biases} biases detected and mitigated")
        
        print(f"Saved to: {output_file.name}")
        
        # Add delay between processing (for batch mode)
        if delay_seconds > 0:
            print(f"Waiting {delay_seconds} seconds before next file...")
            time.sleep(delay_seconds)
        
    except Exception as e:
        error_msg = str(e)
        result_info["error"] = error_msg
        print(f"❌ Error processing {cv_file.name}: {error_msg}")
        logging.error(f"Error processing {cv_file}: {error_msg}")
    
    finally:
        result_info["processing_time"] = time.time() - start_time
    
    return result_info


def print_batch_summary(results: List[Dict[str, Any]], total_time: float):
    """Print summary of batch processing results"""
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    
    print("\n" + "=" * 70)
    print("📊 BATCH PROCESSING SUMMARY")
    print("=" * 70)
    print(f"📁 Total files processed: {len(results)}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"⏱️  Total time: {total_time:.1f} seconds")
    print(f"📈 Average time per file: {total_time/len(results):.1f} seconds")
    
    if successful > 0:
        print(f"\n💾 Results saved in: results/")
        print("📋 Successful files:")
        
        # Collect scoring insights for summary
        scores = []
        potential_scores = []
        high_potential_count = 0
        bias_detected_count = 0
        
        for result in results:
            if result["success"]:
                print(f"   • {Path(result['file']).name} → {result['candidate_name']}")
                
                # Try to load and analyze the result file for insights
                try:
                    result_file = Path("results") / result.get("output_file", "")
                    if result_file.exists():
                        with open(result_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        evaluation = data.get('evaluation', {}) or {}
                        scoring_details = evaluation.get('scoring_details', {}) or {}
                        
                        # Collect metrics
                        current_score = evaluation.get('suitability_score', 0)
                        potential_score = scoring_details.get('potential_score', current_score)
                        scores.append(current_score)
                        potential_scores.append(potential_score)
                        
                        if potential_score > current_score + 10:  # Significant potential
                            high_potential_count += 1
                        
                        bias_detection = scoring_details.get('bias_detection', {})
                        if bias_detection.get('bias_detected', False):
                            bias_detected_count += 1
                            
                except Exception:
                    pass  # Skip if can't read file
        
        # Show enhanced insights if we have data
        if scores:
            avg_score = sum(scores) / len(scores)
            avg_potential = sum(potential_scores) / len(potential_scores)
            max_score = max(scores)
            
            print(f"\nSCORING INSIGHTS:")
            print(f"   Average Score: {avg_score:.1f}/100")
            print(f"   Average Potential: {avg_potential:.1f}/100")
            print(f"   Highest Score: {max_score}/100")
            print(f"   High Potential Candidates: {high_potential_count}")
            if bias_detected_count > 0:
                print(f"   Bias Detection: {bias_detected_count} candidates had biases detected and mitigated")
    
    if failed > 0:
        print(f"\n❌ Failed files:")
        for result in results:
            if not result["success"]:
                print(f"   • {Path(result['file']).name}: {result['error']}")


def parse_arguments() -> Dict[str, Any]:
    """Parse command line arguments"""
    args = {
        'mode': 'single',  # 'single' or 'batch'
        'input_path': None,
        'position_description': None,
        'delay_seconds': 2,
        'debug_mode': False
    }
    
    if len(sys.argv) < 3:
        return args  # Will trigger usage display
    
    # Parse arguments (skip script name)
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--directory" and i + 1 < len(sys.argv):
            args['mode'] = 'batch'
            args['input_path'] = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--delay" and i + 1 < len(sys.argv):
            try:
                args['delay_seconds'] = int(sys.argv[i + 1])
                i += 2
            except ValueError:
                print("❌ Error: Delay must be a number")
                sys.exit(1)
        elif sys.argv[i] == "--debug":
            args['debug_mode'] = True
            i += 1
        elif not args['input_path'] and not sys.argv[i].startswith("--"):
            # First non-flag argument is input path
            args['input_path'] = sys.argv[i]
            i += 1
        elif not args['position_description'] and not sys.argv[i].startswith("--"):
            # Second non-flag argument is position description
            args['position_description'] = sys.argv[i]
            i += 1
        else:
            i += 1
    
    return args


def main() -> int:
    """Main application entry point"""
    
    # Parse arguments
    args = parse_arguments()
    
    # Display usage if insufficient arguments
    if not args['input_path'] or not args['position_description']:
        print("Usage: python nova.py <cv_file_or_directory> \"<position_description>\" [options]")
        print("       python nova.py --directory <cv_directory> \"<position_description>\" [options]")
        print("\nOptions:")
        print("  --directory <path>    Process all CVs in directory (batch mode)")
        print("  --delay <seconds>    Delay between files in batch mode (default: 2)")
        print("  --debug              Enable debug logging")
        print("\nExamples:")
        print("  # Single file analysis")
        print("  python nova.py resume.pdf \"Senior Python Developer with 5+ years experience\"")
        print("  # Batch directory processing")
        print("  python nova.py --directory ./cvs \"Senior Python Developer with 5+ years experience\"")
        print("  # Batch with custom delay")
        print("  python nova.py --directory ./cvs \"Data Scientist role\" --delay 5 --debug")
        return 1
    
    # Validate position description
    if len(args['position_description'].strip()) < 10:
        print("❌ Error: Position description too short (minimum 10 characters)")
        return 1
    
    # Setup logging
    setup_logging(args['debug_mode'], args['mode'] == 'batch')
    logger = logging.getLogger(__name__)
    
    start_time = time.time()
    
    try:
        print_progress_header(args['mode'] == 'batch')
        
        # Validate input path
        print("🔍 Validating inputs...")
        input_path = Path(args['input_path'])
        
        if args['mode'] == 'single':
            # Single file mode
            if not input_path.exists():
                print(f"❌ Error: File not found: {input_path}")
                return 1
            
            if not FileParser.is_supported_file(str(input_path)):
                print(f"❌ Error: Unsupported file type. Supported: .pdf, .docx, .txt")
                return 1
            
            cv_files = [input_path]
            
        else:
            # Batch mode
            try:
                cv_files = find_cv_files(args['input_path'])
                if not cv_files:
                    print(f"❌ No supported CV files found in {args['input_path']}")
                    print(f"Supported formats: {', '.join(FileParser.SUPPORTED_EXTENSIONS)}")
                    return 1
            except (FileNotFoundError, NotADirectoryError) as e:
                print(f"❌ Error: {str(e)}")
                return 1
        
        print("✅ Input validation passed")
        
        # Load configuration
        print("\n🔧 Loading configuration...")
        config = ConfigLoader.load_nova_config()
        print(f"✅ Configuration loaded - Provider: {config.llm.provider}, Model: {config.llm.model_name}")
        
        # Show rate limiting info
        print_rate_limit_info()
        
        # Initialize Nova
        print("\n🤖 Initializing Nova HR Assistant...")
        nova_graph = NovaGraph(config)
        print("✅ Nova initialized successfully")
        
        # Process files
        if args['mode'] == 'single':
            # Single file processing
            print(f"\n📝 Position: {args['position_description']}")
            print(f"\n🚀 Starting CV analysis...")
            
            if args['debug_mode']:
                print("🐛 Debug mode enabled - detailed logging active")
            
            result = process_single_cv(
                cv_files[0], 
                args['position_description'], 
                nova_graph
            )
            
            if result['success']:
                # Print detailed summary for single file
                analysis_result = json.load(open(result['output_file'], 'r'))
                print_final_summary(analysis_result, result['processing_time'])
            else:
                print(f"\n❌ Analysis failed: {result['error']}")
                return 1
            
        else:
            # Batch processing
            print(f"\n📁 Directory: {args['input_path']}")
            print(f"📝 Position: {args['position_description']}")
            print(f"⏱️  Delay between files: {args['delay_seconds']} seconds")
            print(f"✅ Found {len(cv_files)} CV files to process")
            
            # Create results directory
            results_dir = create_results_directory()
            print(f"📂 Results will be saved to: {results_dir}")
            
            # Process each CV file
            print(f"\n🚀 Starting batch processing...")
            results = []
            
            for i, cv_file in enumerate(cv_files, 1):
                print(f"\n[{i}/{len(cv_files)}] Processing: {cv_file.name}")
                
                try:
                    result = process_single_cv(
                        cv_file, 
                        args['position_description'], 
                        nova_graph, 
                        results_dir,
                        args['delay_seconds']
                    )
                    results.append(result)
                    
                except KeyboardInterrupt:
                    print(f"\n⏹️  Processing interrupted by user")
                    break
                except Exception as e:
                    error_result = {
                        "file": str(cv_file),
                        "success": False,
                        "candidate_name": "unknown",
                        "output_file": None,
                        "error": str(e),
                        "processing_time": 0
                    }
                    results.append(error_result)
                    print(f"❌ Unexpected error: {str(e)}")
            
            # Print batch summary
            total_time = time.time() - start_time
            print_batch_summary(results, total_time)
        
        print("\n" + "=" * 70)
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted by user")
        return 130
        
    except Exception as e:
        print(f"\n❌ Unexpected Error: {str(e)}")
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
