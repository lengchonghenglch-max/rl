"""Create a mock run, approve its confirmations, inspect results, and export a report."""

from rf_agent.services.workflow import workflow


def main() -> None:
    run = workflow.create_run("设计一个 2.4 GHz 带通滤波器，优化插入损耗和回波损耗")
    print(f"1. 需求解析与 mock 检索完成: {run['run_id']} ({run['data_kind']})")
    run = workflow.resolve_approval(run["run_id"], "approve", "演示确认参数范围", "demo-approval-key-001")
    print(f"2. 工程确认并生成 {len(run['sample_batch']['candidates'])} 个 LHS 点")
    name, content, _ = workflow.mock_artifact()
    artifact = workflow.ingest(run["run_id"], name, content)
    print(f"3. 导入 mock S 参数: {artifact['rows']} 行，SHA-256 {artifact['sha256']}")
    print(f"4. 报告已生成: {len(workflow.report(run['run_id']))} bytes")
    print(f"5. 后续页面: /runs/{run['run_id']} → /results → /report")


if __name__ == "__main__":
    main()
