# Anthropic Interview Questions

Source: https://www.1point3acres.com/interview/problems/company/anthropic

> **Note:** Full problem details require login to 1point3acres.com. This document contains publicly available summaries and structure.

---

## Table of Contents

### System Design
1. [Design a 1-to-1 Chat System](#1-design-a-1-to-1-chat-system)
2. [Inference API System Design](#8-inference-api-system-design)
3. [Prompt Playground System Design](#9-prompt-playground-system-design)
4. [Distributed Model Deployment System Design](#12-distributed-model-deployment-system-design)
5. [LLM Request Batching API System Design](#17-llm-request-batching-api-system-design)

### Online Assessments (OA)
6. [Task Management System](#2-task-management-system-online-assessment)
7. [Banking System](#3-banking-system-online-assessment)
8. [Cloud Storage System](#4-cloud-storage-system-online-assessment)
9. [Employee Management System](#7-employee-management-system-online-assessment)
10. [Recipe Manager](#10-recipe-manager-online-assessment)
11. [In-memory Database](#20-in-memory-database-online-assessment)

### Coding Problems
12. [Web Crawler](#5-web-crawler)
13. [LRU Cache (Python)](#11-lru-cache-python)
14. [Deduplicate Files](#13-deduplicate-files)
15. [Batch Image Processor](#15-batch-image-processor)
16. [Tokenize (Python)](#16-tokenize-python)
17. [Converting Stack Samples to Trace Events](#18-converting-stack-samples-to-trace-events)
18. [Distributed Mode and Median](#19-distributed-mode-and-median)

### Behavioral
19. [Culture & Behavioral Interview Questions](#6-culture--behavioral-interview-questions)
20. [Hiring Manager Interview Questions](#14-hiring-manager-interview-questions)

---

## 1. Design a 1-to-1 Chat System

**Type:** System Design

**Problem:** Design a chat system that supports only 1-to-1 messaging between users.

**Structure (5 Phases):**
- Phase 1: Define the Goals (~5 minutes)
- Phase 2: Database Schema & Entities (~5 minutes)
- Phase 3: How Client and Server Talk (~5 minutes)
- Phase 4: System Architecture (~15-25 minutes)
- Phase 5: Handling Scale & Challenges (~15-20 minutes)

**Key Areas:**
- Functional and non-functional requirements
- Data model design
- Real-time communication protocols
- Scalability considerations
- Edge cases and challenges

---

## 2. Task Management System (Online Assessment)

**Type:** Online Assessment (OA)

**Problem:** Implement a task management system.

**Levels:**
- Level 1: Basics - Foundational task management features
- Level 2: Search & Sort - Query and organization capabilities
- Level 3: Users & Assignments - Multi-user functionality
- Level 4: Completion & History - Tracking and audit trails

---

## 3. Banking System (Online Assessment)

**Type:** Online Assessment (OA)

**Problem:** Implement an in-memory banking system.

**Levels:**
- Level 1: Basic Actions
- Level 2: Ranking Spenders
- Level 3: Payments and Cashback
- Level 4: Merging and History

**Additional Sections:**
- Special Rules & Edge Cases
- System Constraints

---

## 4. Cloud Storage System (Online Assessment)

**Type:** Online Assessment (OA)

**Problem:** Implement a simple cloud storage system that maps objects (files) to their metainformation.

**Parts:**
- Part 1: Basic File Management - Core file operations
- Part 2: Searching for Files - Query and retrieval functionality
- Part 3: Managing Users and Storage Limits - Multi-user support with quotas
- Part 4: Compressing Files - File compression features

**Additional Sections:**
- Problem Summary
- Data Format specifications

---

## 5. Web Crawler

**Type:** Coding Problem

**Problem:** Given a starting URL `startUrl` and an interface `HtmlParser` that can fetch all URLs from a given web page, implement a web crawler returning all URLs reachable from the starting point that share the same hostname.

**Sections:**
- Problem Statement
- Part 2: Multithreading (Important!)
- Solution: JavaScript Implementation
- System Design Questions

---

## 6. Culture & Behavioral Interview Questions

**Type:** Behavioral Interview

**Focus:** Anthropic places heavy emphasis on culture fit, AI safety values, and how candidates think about ethical considerations.

**Topic Areas:**
1. Introduction
2. Views on AI Safety and Company Mission
3. Handling Feedback and Disagreements
4. Standing Up for Your Beliefs
5. Interest in Anthropic
6. Life Goals and Personal Changes
7. Final Advice

---

## 7. Employee Management System (Online Assessment)

**Type:** Online Assessment (OA)

**Problem:** Implement a simplified employee management system.

**Parts:**
- Part 1: Basic Features
- Part 2: Tracking Time
- Part 3: Raises and Pay Checks
- Part 4: Bonus Pay Periods

---

## 8. Inference API System Design

**Type:** System Design

**Problem:** Design a high-concurrency inference API system that can handle massive concurrent requests efficiently.

**Structure:**
- Step 1: Defining the Scope
- Step 2: Estimating Scale and Capacity
- Step 3: Designing the API
- Step 4: Database and Data Structure
- Step 5: System Overview
- Step 6: Deep Dive into Key Components
- Step 7: Finding and Fixing Weak Spots

**Additional Sections:**
- Problem Requirements
- Sample Solution
- Extra Discussion Points
- Mistakes to Avoid
- How to Pass the Interview
- Practice Questions
- Study Materials

---

## 9. Prompt Playground System Design

**Type:** System Design

**Problem:** Design a prompt engineering playground similar to ChatGPT Playground or Anthropic Console.

**Structure:**
- Step 1: Defining the Requirements
- Step 2: Estimating Scale and Costs
- Step 3: API Definition
- Step 4: Database Schema Design
- Step 5: Architecture Overview
- Step 6: Deep Dive into Key Components
- Step 7: Fixing Performance Issues

**Additional Sections:**
- The Design Problem
- Additional Design Details
- Comparing Different Approaches
- Interview Advice

---

## 10. Recipe Manager (Online Assessment)

**Type:** Online Assessment (OA)

**Problem:** Implement an in-memory recipe management system.

**Levels:**
- Level 1: Basic Operations
- Level 2: Finding and Organizing Data
- Level 3: Adding Users
- Level 4: History and Rollbacks

**Additional Sections:**
- Critical Rules
- System Limitations
- Code Solution

---

## 11. LRU Cache (Python)

**Type:** Coding Problem

**Problem:** You are given an existing in-memory LRU (Least Recently Used) cache implementation in Python.

**Parts:**
- The Problem - Understanding the existing implementation
- Part 2: Saving Data to Disk
- Extra Questions

---

## 12. Distributed Model Deployment System Design

**Type:** System Design

**Problem:** Design a system that efficiently downloads and distributes a large ML model (e.g., 500GB) from external storage to all GPU workers in a data center cluster.

**Structure:**
- The Challenge
- Proposed Solution
- Understanding the Requirements
- Mathematical analysis
- API Design
- Data Structure Design
- System Architecture
- Deep Dive analysis
- Bottleneck identification and fixes
- Interview Tips

---

## 13. Deduplicate Files

**Type:** Coding Problem

**Problem:** Given a root folder/directory, find all duplicate files within it.

**Sections:**
- Problem Requirements
- Faster Solution
- Follow-up Questions

---

## 14. Hiring Manager Interview Questions

**Type:** Behavioral Interview

**Focus:** Anthropic's hiring manager interview emphasizes project deep dives, technical leadership, collaboration skills, and career alignment.

**Topic Areas:**
1. Summary - Introduction section
2. Technical Experience & Projects - Deep dives into candidate's work
3. Working with Others - Collaboration and interpersonal skills
4. Leading and Teaching - Leadership and mentoring capabilities
5. Your Goals and Work Style - Career aspirations and work preferences
6. Key Advice for Success - Interview tips

---

## 15. Batch Image Processor

**Type:** Coding Problem

**Problem:** Build a batch image processing system.

**Sections:**
- Introduction/problem context
- Files and folders specification
- How to modify images
- Specific requirements and tasks
- Function signatures
- Example test cases
- Interview follow-up questions

---

## 16. Tokenize (Python)

**Type:** Coding Problem

**Problem:** You are given two functions: `tokenize` and `detokenize`.

**Parts:**
- Part 1: Understanding the Code
- Part 2: Reviewing a Proposed Fix
- Part 3: Writing a Better Solution
- Part 4: Follow-Up Questions

---

## 17. LLM Request Batching API System Design

**Type:** System Design

**Problem:** Design an HTTP API that exposes a batch processing function for large language model inference.

**Structure:**
- The Challenge (problem introduction)
- Sample Solution (reference implementation)
- Step 1: Clarifying the Requirements
- Step 2: Estimating Scale and Resources
- Step 3: API Design
- Step 4: Data Storage
- Step 5: Basic System Architecture
- Step 6: Deep Dive into Components
- Step 7: Fixing Potential Problems

---

## 18. Converting Stack Samples to Trace Events

**Type:** Coding Problem

**Problem:** A sampling profiler periodically records the full call stack at a timestamp.

**Sections:**
- Problem Statement
- Follow-up Question: Reducing Noise
- System Design Discussion

---

## 19. Distributed Mode and Median

**Type:** Coding Problem

**Problem:** A very large dataset is distributed across multiple machines (typically 10 workers). Each machine has a portion of the dataset stored locally. Using pre-built interface functions `send(workerid, data)` and `recv()`, implement distributed algorithms.

**Objectives:**
1. Finding the Mode - Identify the most frequently occurring element across the distributed dataset
2. Finding the Median - Compute the median value across all machines

**Key Constraints:**
- Multiple worker nodes (~10) hold partitioned data
- Must use provided communication primitives (send/recv functions)
- Very large dataset implies efficiency considerations

---

## 20. In-memory Database (Online Assessment)

**Type:** Online Assessment (OA)

**Problem:** Implement a simplified version of an in-memory database.

**Levels:**
- Level 1: Core Features - Foundational database operations
- Level 2: Filtering Data - Query capabilities with conditions
- Level 3: Automatic Expiration (TTL) - Time-to-live functionality for data
- Level 4: Historical Data - Tracking and retrieving past states

---

## Summary by Category

| Category | Count | Questions |
|----------|-------|-----------|
| System Design | 5 | Chat System, Inference API, Prompt Playground, Model Deployment, LLM Batching |
| Online Assessment | 6 | Task Management, Banking, Cloud Storage, Employee Management, Recipe Manager, In-memory DB |
| Coding Problems | 7 | Web Crawler, LRU Cache, Deduplicate Files, Batch Image Processor, Tokenize, Stack Samples, Distributed Mode/Median |
| Behavioral | 2 | Culture & Behavioral, Hiring Manager |

---

*Last updated: 2026-04-02*
