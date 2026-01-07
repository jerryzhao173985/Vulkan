#!/usr/bin/env python3
"""
Telemetry module for tracking latency, throughput, and memory metrics.

This module provides comprehensive telemetry and monitoring capabilities
for ML inference operations on Vulkan compute.

Features:
- Latency tracking with percentile calculations
- Throughput measurement (operations per second)
- Memory usage monitoring
- Metric aggregation and statistics
- Export to multiple formats (JSON, CSV, Prometheus)

Example usage:
    from vulkan_ml_sdk.telemetry import Telemetry, MetricType

    # Create telemetry instance
    telemetry = Telemetry()

    # Record metrics
    with telemetry.measure("inference"):
        # ... inference code ...
        pass

    # Or manually record
    telemetry.record_latency("conv2d", 15.5)
    telemetry.record_throughput("inference", 100.0)
    telemetry.record_memory("model_load", 1024 * 1024)

    # Get statistics
    stats = telemetry.get_statistics("inference")
    print(f"P99 latency: {stats['p99']}ms")

    # Export metrics
    telemetry.export("metrics.json", format="json")
"""

import json
import time
import threading
import statistics
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable


class TelemetryError(Exception):
    """Base exception for telemetry-related errors."""
    pass


class ExportError(TelemetryError):
    """Error during metric export."""
    pass


class MetricType(Enum):
    """Types of metrics that can be tracked."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY = "memory"
    COUNTER = "counter"
    GAUGE = "gauge"


@dataclass
class MetricPoint:
    """
    A single metric data point.

    Attributes:
        name: Name/identifier of the metric.
        value: Numeric value of the metric.
        metric_type: Type of metric (latency, throughput, memory, etc.).
        timestamp: Unix timestamp when metric was recorded.
        tags: Additional key-value tags for the metric.
        unit: Unit of measurement (e.g., "ms", "MB", "ops/s").
    """
    name: str
    value: float
    metric_type: MetricType
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert metric point to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.value,
            "timestamp": self.timestamp,
            "tags": self.tags,
            "unit": self.unit,
        }


@dataclass
class MetricStatistics:
    """
    Statistical summary of a metric series.

    Attributes:
        name: Name of the metric.
        count: Number of data points.
        min: Minimum value.
        max: Maximum value.
        mean: Average value.
        median: Median value.
        stddev: Standard deviation.
        p50: 50th percentile.
        p90: 90th percentile.
        p95: 95th percentile.
        p99: 99th percentile.
        sum: Sum of all values.
        unit: Unit of measurement.
    """
    name: str
    count: int
    min: float
    max: float
    mean: float
    median: float
    stddev: float
    p50: float
    p90: float
    p95: float
    p99: float
    sum: float
    unit: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary."""
        return asdict(self)


@dataclass
class MemorySnapshot:
    """
    Memory usage snapshot.

    Attributes:
        timestamp: Unix timestamp of the snapshot.
        allocated_bytes: Bytes currently allocated.
        peak_bytes: Peak memory usage.
        available_bytes: Available memory.
        utilization: Memory utilization percentage (0-100).
        source: Source of the measurement (e.g., "process", "gpu").
    """
    timestamp: float
    allocated_bytes: int
    peak_bytes: int
    available_bytes: int
    utilization: float
    source: str = "process"

    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary."""
        return asdict(self)

    @property
    def allocated_mb(self) -> float:
        """Allocated memory in megabytes."""
        return self.allocated_bytes / (1024 * 1024)

    @property
    def peak_mb(self) -> float:
        """Peak memory in megabytes."""
        return self.peak_bytes / (1024 * 1024)


class MetricSeries:
    """
    A time series of metric data points.

    Manages a collection of metric points for a single named metric,
    providing aggregation and statistical analysis.
    """

    def __init__(
        self,
        name: str,
        metric_type: MetricType,
        unit: str = "",
        max_points: int = 10000
    ):
        """
        Initialize a metric series.

        Args:
            name: Name of the metric.
            metric_type: Type of metric.
            unit: Unit of measurement.
            max_points: Maximum number of points to retain.
        """
        self.name = name
        self.metric_type = metric_type
        self.unit = unit
        self.max_points = max_points
        self._points: List[MetricPoint] = []
        self._lock = threading.Lock()

    def add(
        self,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        timestamp: Optional[float] = None
    ) -> MetricPoint:
        """
        Add a data point to the series.

        Args:
            value: Metric value.
            tags: Optional tags for the point.
            timestamp: Optional timestamp (defaults to now).

        Returns:
            The created MetricPoint.
        """
        point = MetricPoint(
            name=self.name,
            value=value,
            metric_type=self.metric_type,
            timestamp=timestamp or time.time(),
            tags=tags or {},
            unit=self.unit,
        )

        with self._lock:
            self._points.append(point)
            # Trim if exceeding max points
            if len(self._points) > self.max_points:
                self._points = self._points[-self.max_points:]

        return point

    def get_values(self) -> List[float]:
        """Get all values in the series."""
        with self._lock:
            return [p.value for p in self._points]

    def get_points(self) -> List[MetricPoint]:
        """Get all points in the series."""
        with self._lock:
            return list(self._points)

    def get_statistics(self) -> MetricStatistics:
        """
        Calculate statistics for the series.

        Returns:
            MetricStatistics with calculated values.
        """
        with self._lock:
            values = [p.value for p in self._points]

        if not values:
            return MetricStatistics(
                name=self.name,
                count=0,
                min=0.0,
                max=0.0,
                mean=0.0,
                median=0.0,
                stddev=0.0,
                p50=0.0,
                p90=0.0,
                p95=0.0,
                p99=0.0,
                sum=0.0,
                unit=self.unit,
            )

        sorted_values = sorted(values)
        count = len(values)

        return MetricStatistics(
            name=self.name,
            count=count,
            min=min(values),
            max=max(values),
            mean=statistics.mean(values),
            median=statistics.median(values),
            stddev=statistics.stdev(values) if count > 1 else 0.0,
            p50=self._percentile(sorted_values, 50),
            p90=self._percentile(sorted_values, 90),
            p95=self._percentile(sorted_values, 95),
            p99=self._percentile(sorted_values, 99),
            sum=sum(values),
            unit=self.unit,
        )

    def _percentile(self, sorted_values: List[float], percentile: float) -> float:
        """Calculate percentile from sorted values."""
        if not sorted_values:
            return 0.0
        k = (len(sorted_values) - 1) * (percentile / 100.0)
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_values) else f
        if f == c:
            return sorted_values[f]
        return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)

    def clear(self) -> None:
        """Clear all data points."""
        with self._lock:
            self._points.clear()

    def __len__(self) -> int:
        """Return number of data points."""
        with self._lock:
            return len(self._points)


class Telemetry:
    """
    Main telemetry class for tracking and exporting metrics.

    Provides comprehensive telemetry capabilities including latency tracking,
    throughput measurement, memory monitoring, and metric export.

    Example:
        telemetry = Telemetry()

        # Record latency
        telemetry.record_latency("model_inference", 25.5)

        # Use context manager for automatic timing
        with telemetry.measure("preprocessing"):
            # ... code to measure ...
            pass

        # Get statistics
        stats = telemetry.get_statistics("model_inference")
        print(f"Average latency: {stats['mean']}ms")

        # Export to JSON
        telemetry.export("metrics.json", format="json")
    """

    # Supported export formats
    SUPPORTED_FORMATS = ["json", "csv", "prometheus"]

    def __init__(
        self,
        name: str = "default",
        max_points_per_metric: int = 10000,
        auto_memory_tracking: bool = False
    ):
        """
        Initialize the Telemetry instance.

        Args:
            name: Name for this telemetry instance.
            max_points_per_metric: Maximum data points per metric series.
            auto_memory_tracking: If True, periodically track memory usage.
        """
        self._name = name
        self._max_points = max_points_per_metric
        self._series: Dict[str, MetricSeries] = {}
        self._memory_snapshots: List[MemorySnapshot] = []
        self._lock = threading.RLock()
        self._created_at = time.time()
        self._callbacks: List[Callable[[MetricPoint], None]] = []

        # Memory tracking
        self._auto_memory_tracking = auto_memory_tracking
        self._memory_tracker_thread: Optional[threading.Thread] = None
        self._stop_memory_tracking = threading.Event()

        if auto_memory_tracking:
            self._start_memory_tracking()

    @property
    def name(self) -> str:
        """Telemetry instance name."""
        return self._name

    @property
    def supported_formats(self) -> List[str]:
        """List of supported export formats."""
        return self.SUPPORTED_FORMATS.copy()

    @property
    def metric_names(self) -> List[str]:
        """List of all tracked metric names."""
        with self._lock:
            return list(self._series.keys())

    def _get_or_create_series(
        self,
        name: str,
        metric_type: MetricType,
        unit: str = ""
    ) -> MetricSeries:
        """Get existing series or create new one."""
        with self._lock:
            if name not in self._series:
                self._series[name] = MetricSeries(
                    name=name,
                    metric_type=metric_type,
                    unit=unit,
                    max_points=self._max_points
                )
            return self._series[name]

    def record_latency(
        self,
        name: str,
        value_ms: float,
        tags: Optional[Dict[str, str]] = None
    ) -> MetricPoint:
        """
        Record a latency measurement.

        Args:
            name: Name/identifier for this latency metric.
            value_ms: Latency value in milliseconds.
            tags: Optional tags for categorization.

        Returns:
            The recorded MetricPoint.
        """
        series = self._get_or_create_series(name, MetricType.LATENCY, "ms")
        point = series.add(value_ms, tags)
        self._notify_callbacks(point)
        return point

    def record_throughput(
        self,
        name: str,
        ops_per_second: float,
        tags: Optional[Dict[str, str]] = None
    ) -> MetricPoint:
        """
        Record a throughput measurement.

        Args:
            name: Name/identifier for this throughput metric.
            ops_per_second: Operations per second.
            tags: Optional tags for categorization.

        Returns:
            The recorded MetricPoint.
        """
        series = self._get_or_create_series(name, MetricType.THROUGHPUT, "ops/s")
        point = series.add(ops_per_second, tags)
        self._notify_callbacks(point)
        return point

    def record_memory(
        self,
        name: str,
        bytes_used: int,
        tags: Optional[Dict[str, str]] = None
    ) -> MetricPoint:
        """
        Record a memory usage measurement.

        Args:
            name: Name/identifier for this memory metric.
            bytes_used: Memory usage in bytes.
            tags: Optional tags for categorization.

        Returns:
            The recorded MetricPoint.
        """
        series = self._get_or_create_series(name, MetricType.MEMORY, "bytes")
        point = series.add(float(bytes_used), tags)
        self._notify_callbacks(point)
        return point

    def record_counter(
        self,
        name: str,
        value: float = 1.0,
        tags: Optional[Dict[str, str]] = None
    ) -> MetricPoint:
        """
        Record a counter increment.

        Args:
            name: Name/identifier for this counter.
            value: Value to add to the counter.
            tags: Optional tags for categorization.

        Returns:
            The recorded MetricPoint.
        """
        series = self._get_or_create_series(name, MetricType.COUNTER, "count")
        point = series.add(value, tags)
        self._notify_callbacks(point)
        return point

    def record_gauge(
        self,
        name: str,
        value: float,
        unit: str = "",
        tags: Optional[Dict[str, str]] = None
    ) -> MetricPoint:
        """
        Record a gauge value.

        Args:
            name: Name/identifier for this gauge.
            value: Current gauge value.
            unit: Unit of measurement.
            tags: Optional tags for categorization.

        Returns:
            The recorded MetricPoint.
        """
        series = self._get_or_create_series(name, MetricType.GAUGE, unit)
        point = series.add(value, tags)
        self._notify_callbacks(point)
        return point

    @contextmanager
    def measure(
        self,
        name: str,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Context manager for automatic latency measurement.

        Args:
            name: Name for the latency metric.
            tags: Optional tags for categorization.

        Yields:
            None

        Example:
            with telemetry.measure("inference"):
                result = model.infer(input_data)
        """
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.record_latency(name, elapsed_ms, tags)

    def get_series(self, name: str) -> Optional[MetricSeries]:
        """
        Get a metric series by name.

        Args:
            name: Name of the metric series.

        Returns:
            MetricSeries if found, None otherwise.
        """
        with self._lock:
            return self._series.get(name)

    def get_statistics(self, name: str) -> Dict[str, Any]:
        """
        Get statistics for a metric.

        Args:
            name: Name of the metric.

        Returns:
            Dictionary with statistical values.
        """
        series = self.get_series(name)
        if series is None:
            return {}
        return series.get_statistics().to_dict()

    def get_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all metrics.

        Returns:
            Dictionary mapping metric names to statistics.
        """
        with self._lock:
            return {
                name: series.get_statistics().to_dict()
                for name, series in self._series.items()
            }

    def add_callback(
        self,
        callback: Callable[[MetricPoint], None]
    ) -> None:
        """
        Add a callback to be invoked when metrics are recorded.

        Args:
            callback: Function to call with each MetricPoint.
        """
        self._callbacks.append(callback)

    def remove_callback(
        self,
        callback: Callable[[MetricPoint], None]
    ) -> bool:
        """
        Remove a previously added callback.

        Args:
            callback: Callback function to remove.

        Returns:
            True if callback was removed, False if not found.
        """
        try:
            self._callbacks.remove(callback)
            return True
        except ValueError:
            return False

    def _notify_callbacks(self, point: MetricPoint) -> None:
        """Notify all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(point)
            except Exception:
                pass  # Don't let callback errors affect metrics

    def take_memory_snapshot(
        self,
        source: str = "process"
    ) -> MemorySnapshot:
        """
        Take a memory usage snapshot.

        Args:
            source: Source of measurement (e.g., "process", "gpu").

        Returns:
            MemorySnapshot with current memory state.
        """
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF)
            allocated = usage.ru_maxrss * 1024  # Convert to bytes (macOS reports in KB)
        except (ImportError, AttributeError):
            allocated = 0

        # Try to get available memory
        try:
            import os
            if hasattr(os, 'sysconf'):
                available = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
            else:
                available = 16 * 1024 * 1024 * 1024  # Default 16GB
        except (OSError, ValueError):
            available = 16 * 1024 * 1024 * 1024

        peak = max(allocated, max([s.allocated_bytes for s in self._memory_snapshots], default=0))
        utilization = (allocated / available * 100) if available > 0 else 0.0

        snapshot = MemorySnapshot(
            timestamp=time.time(),
            allocated_bytes=allocated,
            peak_bytes=peak,
            available_bytes=available,
            utilization=utilization,
            source=source,
        )

        with self._lock:
            self._memory_snapshots.append(snapshot)

        return snapshot

    def get_memory_snapshots(self) -> List[MemorySnapshot]:
        """Get all memory snapshots."""
        with self._lock:
            return list(self._memory_snapshots)

    def _start_memory_tracking(self, interval_seconds: float = 1.0) -> None:
        """Start automatic memory tracking."""
        def track_memory():
            while not self._stop_memory_tracking.is_set():
                self.take_memory_snapshot()
                self._stop_memory_tracking.wait(interval_seconds)

        self._memory_tracker_thread = threading.Thread(
            target=track_memory,
            daemon=True,
            name="telemetry_memory_tracker"
        )
        self._memory_tracker_thread.start()

    def _stop_memory_tracking_thread(self) -> None:
        """Stop automatic memory tracking."""
        if self._memory_tracker_thread:
            self._stop_memory_tracking.set()
            self._memory_tracker_thread.join(timeout=2.0)
            self._memory_tracker_thread = None

    def export(
        self,
        path: Union[str, Path],
        format: str = "json"
    ) -> None:
        """
        Export metrics to a file.

        Args:
            path: Path to export file.
            format: Export format ("json", "csv", or "prometheus").

        Raises:
            ExportError: If export fails.
            ValueError: If format is not supported.
        """
        if format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {format}. "
                f"Supported formats: {self.SUPPORTED_FORMATS}"
            )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if format == "json":
                self._export_json(path)
            elif format == "csv":
                self._export_csv(path)
            elif format == "prometheus":
                self._export_prometheus(path)
        except Exception as e:
            raise ExportError(f"Failed to export metrics: {e}")

    def _export_json(self, path: Path) -> None:
        """Export metrics to JSON format."""
        data = {
            "name": self._name,
            "created_at": self._created_at,
            "exported_at": time.time(),
            "metrics": {},
            "memory_snapshots": [s.to_dict() for s in self._memory_snapshots],
        }

        with self._lock:
            for name, series in self._series.items():
                data["metrics"][name] = {
                    "type": series.metric_type.value,
                    "unit": series.unit,
                    "statistics": series.get_statistics().to_dict(),
                    "points": [p.to_dict() for p in series.get_points()],
                }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def _export_csv(self, path: Path) -> None:
        """Export metrics to CSV format."""
        lines = ["timestamp,metric_name,metric_type,value,unit,tags"]

        with self._lock:
            for name, series in self._series.items():
                for point in series.get_points():
                    tags_str = ";".join(f"{k}={v}" for k, v in point.tags.items())
                    line = f"{point.timestamp},{point.name},{point.metric_type.value},{point.value},{point.unit},{tags_str}"
                    lines.append(line)

        with open(path, 'w') as f:
            f.write("\n".join(lines))

    def _export_prometheus(self, path: Path) -> None:
        """Export metrics to Prometheus text format."""
        lines = []

        with self._lock:
            for name, series in self._series.items():
                stats = series.get_statistics()
                safe_name = name.replace(".", "_").replace("-", "_")

                # Add HELP and TYPE
                lines.append(f"# HELP {safe_name} {series.metric_type.value} metric")
                lines.append(f"# TYPE {safe_name} gauge")

                # Add statistics as labeled metrics
                lines.append(f'{safe_name}{{stat="count"}} {stats.count}')
                lines.append(f'{safe_name}{{stat="min"}} {stats.min}')
                lines.append(f'{safe_name}{{stat="max"}} {stats.max}')
                lines.append(f'{safe_name}{{stat="mean"}} {stats.mean}')
                lines.append(f'{safe_name}{{stat="p50"}} {stats.p50}')
                lines.append(f'{safe_name}{{stat="p90"}} {stats.p90}')
                lines.append(f'{safe_name}{{stat="p95"}} {stats.p95}')
                lines.append(f'{safe_name}{{stat="p99"}} {stats.p99}')
                lines.append("")

        with open(path, 'w') as f:
            f.write("\n".join(lines))

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert telemetry data to dictionary.

        Returns:
            Dictionary representation of all telemetry data.
        """
        with self._lock:
            return {
                "name": self._name,
                "created_at": self._created_at,
                "metric_count": len(self._series),
                "metrics": {
                    name: {
                        "type": series.metric_type.value,
                        "unit": series.unit,
                        "count": len(series),
                        "statistics": series.get_statistics().to_dict(),
                    }
                    for name, series in self._series.items()
                },
                "memory_snapshot_count": len(self._memory_snapshots),
            }

    def reset(self) -> None:
        """Clear all metrics and memory snapshots."""
        with self._lock:
            for series in self._series.values():
                series.clear()
            self._series.clear()
            self._memory_snapshots.clear()

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of telemetry state.

        Returns:
            Dictionary with summary information.
        """
        with self._lock:
            total_points = sum(len(s) for s in self._series.values())

            return {
                "name": self._name,
                "metric_count": len(self._series),
                "total_data_points": total_points,
                "memory_snapshots": len(self._memory_snapshots),
                "supported_formats": self.supported_formats,
                "uptime_seconds": time.time() - self._created_at,
            }

    def close(self) -> None:
        """
        Close the telemetry instance and clean up resources.

        Should be called when done using telemetry to stop
        background threads and release resources.
        """
        self._stop_memory_tracking_thread()
        self._callbacks.clear()

    def __enter__(self) -> "Telemetry":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - close telemetry."""
        self.close()

    def __repr__(self) -> str:
        return (
            f"Telemetry(name={self._name!r}, "
            f"metrics={len(self._series)}, "
            f"formats={self.supported_formats})"
        )


class TelemetryAggregator:
    """
    Aggregates metrics from multiple Telemetry instances.

    Useful for combining telemetry from multiple sources
    or for distributed inference scenarios.

    Example:
        aggregator = TelemetryAggregator()
        aggregator.add_source(telemetry1)
        aggregator.add_source(telemetry2)

        combined_stats = aggregator.get_combined_statistics("inference_latency")
    """

    def __init__(self):
        """Initialize the aggregator."""
        self._sources: List[Telemetry] = []
        self._lock = threading.Lock()

    def add_source(self, telemetry: Telemetry) -> None:
        """
        Add a telemetry source.

        Args:
            telemetry: Telemetry instance to add.
        """
        with self._lock:
            if telemetry not in self._sources:
                self._sources.append(telemetry)

    def remove_source(self, telemetry: Telemetry) -> bool:
        """
        Remove a telemetry source.

        Args:
            telemetry: Telemetry instance to remove.

        Returns:
            True if removed, False if not found.
        """
        with self._lock:
            try:
                self._sources.remove(telemetry)
                return True
            except ValueError:
                return False

    def get_combined_statistics(self, metric_name: str) -> Dict[str, Any]:
        """
        Get combined statistics for a metric across all sources.

        Args:
            metric_name: Name of the metric.

        Returns:
            Combined statistics dictionary.
        """
        all_values = []

        with self._lock:
            for source in self._sources:
                series = source.get_series(metric_name)
                if series:
                    all_values.extend(series.get_values())

        if not all_values:
            return {}

        sorted_values = sorted(all_values)
        count = len(all_values)

        def percentile(p: float) -> float:
            k = (count - 1) * (p / 100.0)
            f = int(k)
            c = f + 1 if f + 1 < count else f
            if f == c:
                return sorted_values[f]
            return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)

        return {
            "name": metric_name,
            "source_count": len(self._sources),
            "count": count,
            "min": min(all_values),
            "max": max(all_values),
            "mean": statistics.mean(all_values),
            "median": statistics.median(all_values),
            "stddev": statistics.stdev(all_values) if count > 1 else 0.0,
            "p50": percentile(50),
            "p90": percentile(90),
            "p95": percentile(95),
            "p99": percentile(99),
            "sum": sum(all_values),
        }

    def get_all_metric_names(self) -> List[str]:
        """
        Get all unique metric names across all sources.

        Returns:
            List of unique metric names.
        """
        names = set()
        with self._lock:
            for source in self._sources:
                names.update(source.metric_names)
        return sorted(names)

    def export_all(
        self,
        directory: Union[str, Path],
        format: str = "json"
    ) -> List[Path]:
        """
        Export all sources to a directory.

        Args:
            directory: Directory to export to.
            format: Export format.

        Returns:
            List of exported file paths.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        exported = []

        with self._lock:
            for i, source in enumerate(self._sources):
                filename = f"{source.name or f'source_{i}'}.{format}"
                path = directory / filename
                source.export(path, format=format)
                exported.append(path)

        return exported

    def __len__(self) -> int:
        """Return number of sources."""
        return len(self._sources)

    def __repr__(self) -> str:
        return f"TelemetryAggregator(sources={len(self._sources)})"
