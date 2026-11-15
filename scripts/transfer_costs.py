"""Read app-scoped Modal metered usage; requires the authenticated Modal SDK.

Modal's billing endpoint is an internal SDK API. Preserve the SDK version and raw
records; the result is metered usage, not an invoice or proof that recent usage settled.
"""
import argparse
import asyncio
import json
from datetime import UTC, datetime
from decimal import Decimal
from importlib.metadata import version
from pathlib import Path


def utc_timestamp(value):
  parsed = datetime.fromisoformat(value)
  return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


async def run(args):
  from google.protobuf.json_format import MessageToDict
  from modal.client import _Client
  from modal_proto import api_pb2

  output = Path(args.output)
  if output.exists():
    raise ValueError('Preserve the prior billing snapshot; choose a distinct output path')
  ids = sorted({row['app_id'] for row in json.loads(Path(args.ledger).read_text())})
  if not ids:
    raise ValueError('Empty app ledger')
  start, end = utc_timestamp(args.start), utc_timestamp(args.end)
  if start >= end:
    raise ValueError('Billing interval must have start before end')
  request = api_pb2.WorkspaceBillingReportRequest(resolution='h', app_ids=ids)
  request.start_timestamp.FromDatetime(start)
  request.end_timestamp.FromDatetime(end)
  client = await _Client.from_env()
  records = [MessageToDict(row, preserving_proto_field_name=True)
             async for row in client.stub.WorkspaceBillingReport.unary_stream(request)]
  records = [row for row in records if row.get('object_id') in ids]
  report = dict(retrieved_at_utc=datetime.now(UTC).isoformat(), modal_sdk_version=version('modal'),
                start_utc=start.isoformat(), end_utc=end.isoformat(), requested_app_ids=ids, records=records,
                total_usd=str(sum((Decimal(row['cost']) for row in records), Decimal(0))),
                limitation='Metered hourly records only; recent usage may not be settled. Not an invoice.')
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(report, indent=2)+'\n')
  print(json.dumps({k: v for k, v in report.items() if k not in ('records', 'requested_app_ids')}, indent=2))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--ledger', required=True)
  parser.add_argument('--start', required=True, help='UTC date or timestamp, e.g. 2026-11-13')
  parser.add_argument('--end', required=True, help='Exclusive UTC date or timestamp')
  parser.add_argument('--output', required=True)
  asyncio.run(run(parser.parse_args()))
