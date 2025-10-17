type QuickAction = {
  label: string
  description?: string
}

type QuickActionsProps = {
  items: QuickAction[]
}

const QuickActions = ({ items }: QuickActionsProps) => {
  return (
    <nav className="quick-actions" aria-label="快捷问题建议">
      {items.map(action => (
        <button className="quick-action" type="button" key={action.label}>
          <span>{action.label}</span>
          {action.description ? <small>{action.description}</small> : null}
        </button>
      ))}
    </nav>
  )
}

export default QuickActions
