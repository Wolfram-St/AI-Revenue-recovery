interface MetricCardProps {
  label: string;
  value: string | number;
  subtitle?: string;
  color?: 'emerald' | 'amber' | 'red' | 'blue';
}

const colorMap = {
  emerald: 'text-emerald-400',
  amber: 'text-amber-400',
  red: 'text-red-400',
  blue: 'text-blue-400',
};

export default function MetricCard({ label, value, subtitle, color = 'emerald' }: MetricCardProps) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
      <p className="text-xs text-gray-500 uppercase tracking-wide mb-1">{label}</p>
      <p className={`text-2xl font-bold ${colorMap[color]}`}>{value}</p>
      {subtitle && <p className="text-xs text-gray-500 mt-1">{subtitle}</p>}
    </div>
  );
}
