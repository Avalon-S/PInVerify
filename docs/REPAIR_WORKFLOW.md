# PIN-v2 Dataset Repair Workflow

```mermaid
graph TD
    %% Define Styles
    classDef process fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef decision fill:#fff9c4,stroke:#fbc02d,stroke-width:2px,stroke-dasharray: 5 5;
    classDef data fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px;
    classDef startend fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,rx:10,ry:10;

    Start((Start)):::startend --> Init[Initialize Work Root<br/>& Match Pool]:::process
    Init --> CheckCondition{Bad Episodes > 0?}:::decision

    subgraph "Iterative Repair Loop"
        direction TB
        Backup[<b>Step 1: Backup</b><br/>Snapshot current content<br/><i>(content_v0, v1...)</i>]:::process
        
        Repair[<b>Step 2: Repair</b><br/>Generate Repaired Dataset<br/><i>(Sample from Match Pool)</i>]:::process
        
        Swap[<b>Step 3: Swap</b><br/>Replace Dataset Content]:::process
        
        Verify[<b>Step 4: Verification</b><br/>Parallel Execution<br/><i>(eval_goalview.py)</i>]:::process
        
        Analyze[<b>Step 5: Analysis</b><br/>Classify Results<br/><i>(classify_abnormal_episodes.py)</i>]:::process
        
        Update[Update Bad List<br/>& Statistics]:::data
    end

    CheckCondition -- Yes --> Backup
    Backup --> Repair
    Repair --> Swap
    Swap --> Verify
    Verify --> Analyze
    Analyze --> Update
    Update --> CheckCondition

    CheckCondition -- No --> Finish((Finish<br/>Clean Dataset)):::startend

    %% Link Annotations
    linkStyle 6 stroke:#fbc02d,stroke-width:2px;
    linkStyle 12 stroke:#2e7d32,stroke-width:2px;
```
