---
title: "The Training Loop"
description: "The engineering steps needed to turn tokenized data into a verified parameter update."
question: "What must work before it is safe to scale a training run?"
order: 1
status: draft
---

This is a **content-system smoke test**, not a reconstruction of the training experiments. M1.5 will replace it with a sourced chapter.

## The engineering problem

Training requires the data loader, forward pass, loss, backward pass and optimizer to agree on what an update means. A single working step is only the beginning of that verification.

## Why recovery matters

If training must resume, the state extends beyond model weights: optimizer state, data position and random generators can affect continuation.

## What comes next

The finished chapter will connect the actual implementation and the validation evidence to each engineering decision. See the [Training index](/training/) for the full sequence.
