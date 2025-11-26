"use client";

type Props = {
  strategies: string[];
  selected: string;
  onChange: (strategy: string) => void;
};

export default function StrategySelector({
  strategies,
  selected,
  onChange,
}: Props) {
  return (
    <label className="flex items-center gap-2 text-sm">
      <span>Strategy:</span>
      <select
        value={selected}
        onChange={(e) => onChange(e.target.value)}
        className="border px-2 py-1 rounded"
      >
        {strategies.map((strategy) => (
          <option key={strategy} value={strategy}>
            {strategy}
          </option>
        ))}
      </select>
    </label>
  );
}
