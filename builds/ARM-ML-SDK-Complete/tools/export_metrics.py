#!/usr/bin/env python3
"""
JSON metrics exporter for profiling data with aggregation and filtering capabilities
"""

import json
import sys
import os
import glob
from datetime import datetime
from typing import List, Dict, Any, Optional


class MetricsExporter:
    """Export and aggregate profiling metrics to JSON format"""

    def __init__(self):
        self.metrics = []
        self.raw_data = []
        self.sources = []

    def load_profile_file(self, filepath: str) -> bool:
        """Load metrics from a profile JSON file"""
        if not os.path.exists(filepath):
            return False

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            self.raw_data.append(data)
            self.sources.append(filepath)

            # Handle different profile formats
            if isinstance(data, dict):
                if "metrics" in data:
                    # Standard profile format
                    self.metrics.extend(data["metrics"])
                elif "summary" in data:
                    # Real-time profile format
                    self.metrics.append({
                        "name": data.get("operation", "unknown"),
                        "time_ms": data["summary"].get("average_time_ms", 0),
                        "min_ms": data["summary"].get("min_time_ms", 0),
                        "max_ms": data["summary"].get("max_time_ms", 0),
                        "iterations": data["summary"].get("total_iterations", 1),
                        "status": "success" if data["summary"].get("successful_runs", 0) > 0 else "failed",
                        "source": filepath
                    })
                else:
                    # Single metric format
                    self.metrics.append(data)
            elif isinstance(data, list):
                # Array of metrics
                self.metrics.extend(data)

            return True
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON from {filepath}: {e}", file=sys.stderr)
            return False
        except Exception as e:
            print(f"Warning: Error loading {filepath}: {e}", file=sys.stderr)
            return False

    def load_directory(self, directory: str, pattern: str = "profile_*.json") -> int:
        """Load all profile files matching pattern from directory"""
        search_path = os.path.join(directory, pattern)
        files = glob.glob(search_path)
        loaded = 0

        for filepath in files:
            if self.load_profile_file(filepath):
                loaded += 1

        return loaded

    def filter_metrics(self,
                       operation: Optional[str] = None,
                       status: Optional[str] = None,
                       min_time: Optional[float] = None,
                       max_time: Optional[float] = None) -> List[Dict]:
        """Filter metrics based on criteria"""
        filtered = self.metrics.copy()

        if operation:
            filtered = [m for m in filtered if operation.lower() in m.get("name", "").lower()]

        if status:
            filtered = [m for m in filtered if m.get("status") == status]

        if min_time is not None:
            filtered = [m for m in filtered if m.get("time_ms", 0) >= min_time]

        if max_time is not None:
            filtered = [m for m in filtered if m.get("time_ms", float('inf')) <= max_time]

        return filtered

    def calculate_statistics(self, metrics: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Calculate aggregate statistics from metrics"""
        if metrics is None:
            metrics = self.metrics

        if not metrics:
            return {"error": "No metrics available"}

        times = [m.get("time_ms", 0) for m in metrics if m.get("time_ms")]
        successful = [m for m in metrics if m.get("status") == "success"]
        failed = [m for m in metrics if m.get("status") == "failed"]

        stats = {
            "total_metrics": len(metrics),
            "successful_count": len(successful),
            "failed_count": len(failed),
            "success_rate": len(successful) / len(metrics) * 100 if metrics else 0
        }

        if times:
            stats.update({
                "average_time_ms": sum(times) / len(times),
                "min_time_ms": min(times),
                "max_time_ms": max(times),
                "total_time_ms": sum(times)
            })

            # Calculate standard deviation
            if len(times) > 1:
                avg = stats["average_time_ms"]
                variance = sum((t - avg) ** 2 for t in times) / len(times)
                stats["std_dev_ms"] = variance ** 0.5

            # Calculate percentiles
            sorted_times = sorted(times)
            n = len(sorted_times)
            stats["p50_time_ms"] = sorted_times[n // 2]
            stats["p90_time_ms"] = sorted_times[int(n * 0.9)] if n >= 10 else sorted_times[-1]
            stats["p99_time_ms"] = sorted_times[int(n * 0.99)] if n >= 100 else sorted_times[-1]

        # Group by operation
        operations = {}
        for m in metrics:
            name = m.get("name", "unknown")
            if name not in operations:
                operations[name] = []
            operations[name].append(m)

        stats["operations"] = {}
        for name, op_metrics in operations.items():
            op_times = [m.get("time_ms", 0) for m in op_metrics if m.get("time_ms")]
            if op_times:
                stats["operations"][name] = {
                    "count": len(op_metrics),
                    "average_time_ms": sum(op_times) / len(op_times),
                    "min_time_ms": min(op_times),
                    "max_time_ms": max(op_times)
                }

        return stats

    def export_json(self,
                    output_path: str,
                    include_stats: bool = True,
                    include_raw: bool = False,
                    pretty: bool = True,
                    metrics: Optional[List[Dict]] = None) -> str:
        """Export metrics to JSON file"""
        if metrics is None:
            metrics = self.metrics

        export_data = {
            "export_info": {
                "generated_at": datetime.now().isoformat(),
                "tool_version": "1.0.0",
                "sources": self.sources,
                "total_metrics": len(metrics)
            },
            "metrics": metrics
        }

        if include_stats:
            export_data["statistics"] = self.calculate_statistics(metrics)

        if include_raw:
            export_data["raw_data"] = self.raw_data

        indent = 2 if pretty else None

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=indent)

        return output_path

    def export_summary(self, output_path: str, pretty: bool = True) -> str:
        """Export only statistics summary to JSON"""
        summary = {
            "export_info": {
                "generated_at": datetime.now().isoformat(),
                "sources": self.sources
            },
            "statistics": self.calculate_statistics()
        }

        indent = 2 if pretty else None

        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=indent)

        return output_path

    def export_streaming(self, output_path: str, metrics: Optional[List[Dict]] = None) -> str:
        """Export metrics as newline-delimited JSON (NDJSON) for streaming"""
        if metrics is None:
            metrics = self.metrics

        with open(output_path, 'w') as f:
            for metric in metrics:
                f.write(json.dumps(metric) + '\n')

        return output_path

    def print_summary(self):
        """Print summary of loaded metrics"""
        stats = self.calculate_statistics()

        print("=== Metrics Export Summary ===")
        print(f"Sources loaded: {len(self.sources)}")
        print(f"Total metrics: {stats.get('total_metrics', 0)}")
        print(f"Successful: {stats.get('successful_count', 0)}")
        print(f"Failed: {stats.get('failed_count', 0)}")

        if "average_time_ms" in stats:
            print(f"\nTiming Statistics:")
            print(f"  Average: {stats['average_time_ms']:.2f} ms")
            print(f"  Min: {stats['min_time_ms']:.2f} ms")
            print(f"  Max: {stats['max_time_ms']:.2f} ms")
            if "std_dev_ms" in stats:
                print(f"  Std Dev: {stats['std_dev_ms']:.2f} ms")
            print(f"  P50: {stats['p50_time_ms']:.2f} ms")
            print(f"  P90: {stats['p90_time_ms']:.2f} ms")

        if stats.get("operations"):
            print(f"\nOperations:")
            for name, op_stats in stats["operations"].items():
                print(f"  {name}: {op_stats['average_time_ms']:.2f} ms avg ({op_stats['count']} samples)")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Export profiling metrics to JSON format with aggregation and filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input profile_conv2d.json --output metrics.json
  %(prog)s --directory ./profiles --output all_metrics.json
  %(prog)s --input profile.json --filter-operation conv2d --output conv2d_metrics.json
  %(prog)s --directory . --summary-only --output summary.json
  %(prog)s --input profile.json --format ndjson --output metrics.ndjson
        """
    )

    parser.add_argument("--input", "-i", action="append", dest="inputs",
                        help="Input profile JSON file (can be specified multiple times)")
    parser.add_argument("--directory", "-d",
                        help="Directory to load profile files from")
    parser.add_argument("--pattern", "-p", default="profile_*.json",
                        help="File pattern for directory loading (default: profile_*.json)")
    parser.add_argument("--output", "-o", required=True,
                        help="Output file path")
    parser.add_argument("--format", "-f", choices=["json", "ndjson", "summary"],
                        default="json", help="Output format (default: json)")
    parser.add_argument("--compact", "-c", action="store_true",
                        help="Output compact JSON without indentation")
    parser.add_argument("--include-raw", action="store_true",
                        help="Include raw source data in export")
    parser.add_argument("--no-stats", action="store_true",
                        help="Exclude statistics from export")
    parser.add_argument("--summary-only", action="store_true",
                        help="Export only summary statistics")
    parser.add_argument("--filter-operation",
                        help="Filter by operation name (substring match)")
    parser.add_argument("--filter-status", choices=["success", "failed"],
                        help="Filter by status")
    parser.add_argument("--filter-min-time", type=float,
                        help="Filter by minimum execution time (ms)")
    parser.add_argument("--filter-max-time", type=float,
                        help="Filter by maximum execution time (ms)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress output messages")

    args = parser.parse_args()

    # Validate inputs
    if not args.inputs and not args.directory:
        parser.print_help()
        print("\nError: Either --input or --directory is required", file=sys.stderr)
        sys.exit(1)

    exporter = MetricsExporter()

    # Load input files
    if args.inputs:
        for input_file in args.inputs:
            if not os.path.exists(input_file):
                print(f"Error: Input file not found: {input_file}", file=sys.stderr)
                sys.exit(1)
            if not exporter.load_profile_file(input_file):
                print(f"Warning: Failed to load {input_file}", file=sys.stderr)

    # Load directory
    if args.directory:
        if not os.path.isdir(args.directory):
            print(f"Error: Directory not found: {args.directory}", file=sys.stderr)
            sys.exit(1)
        loaded = exporter.load_directory(args.directory, args.pattern)
        if not args.quiet:
            print(f"Loaded {loaded} profile file(s) from {args.directory}")

    if not exporter.metrics:
        print("Error: No metrics loaded", file=sys.stderr)
        sys.exit(1)

    # Apply filters
    filtered_metrics = exporter.filter_metrics(
        operation=args.filter_operation,
        status=args.filter_status,
        min_time=args.filter_min_time,
        max_time=args.filter_max_time
    )

    if not filtered_metrics:
        print("Error: No metrics match the filter criteria", file=sys.stderr)
        sys.exit(1)

    # Export based on format
    pretty = not args.compact

    if args.summary_only or args.format == "summary":
        exporter.export_summary(args.output, pretty=pretty)
    elif args.format == "ndjson":
        exporter.export_streaming(args.output, metrics=filtered_metrics)
    else:
        exporter.export_json(
            args.output,
            include_stats=not args.no_stats,
            include_raw=args.include_raw,
            pretty=pretty,
            metrics=filtered_metrics
        )

    if not args.quiet:
        exporter.print_summary()
        print(f"\nExported to: {args.output}")


if __name__ == "__main__":
    main()
